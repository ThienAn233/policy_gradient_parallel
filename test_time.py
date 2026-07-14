import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from deer import deer_alg


# ============================================================
# Timing utilities
# ============================================================

def block_until_ready(tree):
    """Block on every JAX array in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def time_jax(fn, warmup=1, repeat=5):
    """
    Time a JAX function correctly.

    The warmup calls compile the function, so they are not counted.
    """
    for _ in range(warmup):
        out = fn()
        block_until_ready(out)

    times = []
    out = None

    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        block_until_ready(out)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    times = jnp.array(times)
    mean_ms = float(jnp.mean(times))
    std_ms = float(jnp.std(times))

    return out, mean_ms, std_ms


# ============================================================
# Stable random system generator
# ============================================================

def make_stable_matrix(key, n, radius=0.85):
    """
    Generate a random n x n matrix with spectral radius approximately radius.
    """
    M = jr.normal(key, shape=(n, n)) / jnp.sqrt(n)
    eigvals = jnp.linalg.eigvals(M)
    spectral_radius = jnp.max(jnp.abs(eigvals))
    return M * (radius / (spectral_radius + 1e-12))


def make_system(n, m, seed=0, radius=0.85):
    """
    Generate a stable closed-loop linear system.

    State dimension:   n
    Control dimension: m

    Dynamics:
        x_{k+1} = A x_k + B u_k
        u_k     = -K x_k

    Closed-loop:
        x_{k+1} = (A - B K) x_k

    We generate a stable F_cl first, then choose A = F_cl + B K.
    This guarantees that A - B K = F_cl is stable.
    """
    key = jr.PRNGKey(seed)
    k1, k2, k3, k4 = jr.split(key, 4)

    F_cl = make_stable_matrix(k1, n, radius=radius)

    B = jr.normal(k2, shape=(n, m)) / jnp.sqrt(m)
    K = 0.1 * jr.normal(k3, shape=(m, n)) / jnp.sqrt(n)

    A = F_cl + B @ K

    x0 = jnp.ones(n)

    states_guess = jr.normal(k4, shape=(T_max, n))

    dummy_inputs = jnp.zeros((T_max, m))

    return A, B, K, x0, states_guess, dummy_inputs


# ============================================================
# Global experiment settings
# ============================================================

T_max = 300
tol = 1e-7
repeat = 5
warmup = 1
deer_iters = 50

# Try increasing sizes.
# Each pair is (state dimension n, control dimension m).
configs = [
    (3, 1),
    (4, 2),
    (8, 4),
    (16, 8),
    (32, 16),
    # (64, 32),   # uncomment if your machine is strong enough
]


# ============================================================
# One benchmark for one (n, m)
# ============================================================

def run_one_benchmark(n, m, seed=0):
    A, B, K, x0, states_guess, dummy_inputs = make_system(
        n=n,
        m=m,
        seed=seed,
        radius=0.85,
    )

    lambda_T = jnp.zeros(n)

    costate_guess = jr.normal(
        jr.PRNGKey(seed + 1000),
        shape=(T_max, n),
    )

    x_ref = jnp.ones(n)

    # ------------------------------------------------------------
    # Closed-loop dynamics
    # ------------------------------------------------------------

    def f(x, u_dummy):
        """
        Closed-loop dynamics:

            u = -K x
            x_next = A x + B u

        Equivalently:

            x_next = (A - B K) x
        """
        u = -K @ x
        return A @ x + B @ u

    F_x_T = (A - B @ K).T

    # ------------------------------------------------------------
    # Sequential forward rollout
    # ------------------------------------------------------------

    def rollout_step(x, _):
        x_next = f(x, None)
        return x_next, x_next

    def forward_sequential_raw():
        _, states_rollout = jax.lax.scan(
            rollout_step,
            x0,
            jnp.arange(T_max),
        )
        return states_rollout

    forward_sequential = jax.jit(forward_sequential_raw)

    # ------------------------------------------------------------
    # DEER / parallel forward rollout
    # ------------------------------------------------------------

    def forward_deer_raw():
        _, states_deer, newton_steps, *_ = deer_alg(
            f,
            x0,
            states_guess,
            dummy_inputs,
            num_iters=deer_iters,
            full_trace=False,
            Ts=None,
            tol=tol,
        )
        return states_deer, newton_steps

    forward_deer = jax.jit(forward_deer_raw)

    # ------------------------------------------------------------
    # Backward costate dynamics
    # ------------------------------------------------------------

    def backward_costate_step(lambda_next, x_k):
        """
        Costate recursion:

            lambda_k = grad_x l(x_k, K) + F_x^T lambda_{k+1}

        Cost:

            l(x, u) = ||x - x_ref||^2 + ||u||^2

        Policy:

            u = -K x
        """
        u_k = -K @ x_k

        grad_x_l = 2.0 * (x_k - x_ref) - 2.0 * K.T @ u_k

        lambda_k = grad_x_l + F_x_T @ lambda_next

        return lambda_k

    def back_step(lambda_next, x_k):
        lambda_k = backward_costate_step(lambda_next, x_k)
        return lambda_k, lambda_k

    def backward_sequential_given_states_raw(states_rollout):
        """
        Returns costates in reverse order:

            [lambda_{T-1}, ..., lambda_0]
        """
        x_traj = jnp.vstack([x0, states_rollout[:-1]])

        _, lambda_traj_rev = jax.lax.scan(
            back_step,
            lambda_T,
            jnp.flip(x_traj, axis=0),
        )

        return lambda_traj_rev

    backward_sequential_given_states = jax.jit(
        backward_sequential_given_states_raw
    )

    def backward_deer_given_states_raw(states_rollout):
        """
        DEER backward solve.

        Returns costates in reverse order:

            [lambda_{T-1}, ..., lambda_0]
        """
        x_traj = jnp.vstack([x0, states_rollout[:-1]])
        x_traj_rev = jnp.flip(x_traj, axis=0)

        _, costate_deer, newton_steps, *_ = deer_alg(
            backward_costate_step,
            lambda_T,
            costate_guess,
            x_traj_rev,
            num_iters=deer_iters,
            full_trace=False,
            Ts=None,
            tol=tol,
        )

        return costate_deer, newton_steps

    backward_deer_given_states = jax.jit(backward_deer_given_states_raw)

    # ------------------------------------------------------------
    # Combined forward + backward
    # ------------------------------------------------------------

    def both_sequential_raw():
        states_rollout = forward_sequential_raw()
        lambda_traj_rev = backward_sequential_given_states_raw(states_rollout)
        return states_rollout, lambda_traj_rev

    both_sequential = jax.jit(both_sequential_raw)

    def both_deer_raw():
        states_deer, fwd_steps = forward_deer_raw()
        costate_deer, bwd_steps = backward_deer_given_states_raw(states_deer)
        return states_deer, costate_deer, fwd_steps, bwd_steps

    both_deer = jax.jit(both_deer_raw)

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    states_rollout, t_fwd_seq, sd_fwd_seq = time_jax(
        forward_sequential,
        warmup=warmup,
        repeat=repeat,
    )

    forward_deer_out, t_fwd_deer, sd_fwd_deer = time_jax(
        forward_deer,
        warmup=warmup,
        repeat=repeat,
    )

    states_deer, fwd_newton_steps = forward_deer_out

    lambda_seq_rev, t_bwd_seq, sd_bwd_seq = time_jax(
        lambda: backward_sequential_given_states(states_rollout),
        warmup=warmup,
        repeat=repeat,
    )

    backward_deer_out, t_bwd_deer, sd_bwd_deer = time_jax(
        lambda: backward_deer_given_states(states_rollout),
        warmup=warmup,
        repeat=repeat,
    )

    costate_deer, bwd_newton_steps = backward_deer_out

    both_seq_out, t_both_seq, sd_both_seq = time_jax(
        both_sequential,
        warmup=warmup,
        repeat=repeat,
    )

    both_deer_out, t_both_deer, sd_both_deer = time_jax(
        both_deer,
        warmup=warmup,
        repeat=repeat,
    )

    states_seq_both, lambda_seq_both_rev = both_seq_out
    states_deer_both, lambda_deer_both_rev, fwd_steps_both, bwd_steps_both = both_deer_out

    # ------------------------------------------------------------
    # Accuracy checks
    # ------------------------------------------------------------

    forward_error = float(jnp.max(jnp.abs(states_deer - states_rollout)))
    backward_error = float(jnp.max(jnp.abs(costate_deer - lambda_seq_rev)))

    both_forward_error = float(jnp.max(jnp.abs(states_deer_both - states_seq_both)))
    both_backward_error = float(jnp.max(jnp.abs(lambda_deer_both_rev - lambda_seq_both_rev)))

    # ------------------------------------------------------------
    # Policy gradient
    # ------------------------------------------------------------

    lambda_traj = jnp.flip(lambda_seq_rev, axis=0)

    x_traj = jnp.vstack([x0, states_rollout[:-1]])

    lambda_k_plus_1_traj = jnp.vstack([
        lambda_traj[1:],
        lambda_T,
    ])

    def compute_grad_step(carry, inputs):
        x_k, lambda_k_plus_1 = inputs

        u_k = -K @ x_k

        # h_u = 2 R u_k + B^T lambda_{k+1}, with R = I
        h_u = 2.0 * u_k + B.T @ lambda_k_plus_1

        # Since u = -Kx:
        # grad_K J step = - h_u x_k^T
        grad_K_step = -jnp.outer(h_u, x_k)

        return carry, grad_K_step

    _, all_K_grads = jax.lax.scan(
        compute_grad_step,
        None,
        (x_traj, lambda_k_plus_1_traj),
    )

    grad_K = jnp.sum(all_K_grads, axis=0)
    grad_norm = float(jnp.linalg.norm(grad_K))

    return {
        "n": n,
        "m": m,
        "T": T_max,

        "t_fwd_seq": t_fwd_seq,
        "t_fwd_deer": t_fwd_deer,
        "t_bwd_seq": t_bwd_seq,
        "t_bwd_deer": t_bwd_deer,
        "t_both_seq": t_both_seq,
        "t_both_deer": t_both_deer,

        "forward_speedup": t_fwd_seq / t_fwd_deer,
        "backward_speedup": t_bwd_seq / t_bwd_deer,
        "both_speedup": t_both_seq / t_both_deer,

        "forward_error": forward_error,
        "backward_error": backward_error,
        "both_forward_error": both_forward_error,
        "both_backward_error": both_backward_error,

        "fwd_newton_steps": int(fwd_newton_steps),
        "bwd_newton_steps": int(bwd_newton_steps),

        "grad_norm": grad_norm,
    }


# ============================================================
# Run all benchmarks
# ============================================================

results = []

print("\n================ Shape Scaling Benchmark ================\n")

for idx, (n, m) in enumerate(configs):
    print(f"Running benchmark for n={n}, m={m} ...")

    result = run_one_benchmark(
        n=n,
        m=m,
        seed=idx,
    )

    results.append(result)

    print(
        f"  Forward:  seq={result['t_fwd_seq']:.3f} ms, "
        f"DEER={result['t_fwd_deer']:.3f} ms, "
        f"speedup={result['forward_speedup']:.3f}x"
    )

    print(
        f"  Backward: seq={result['t_bwd_seq']:.3f} ms, "
        f"DEER={result['t_bwd_deer']:.3f} ms, "
        f"speedup={result['backward_speedup']:.3f}x"
    )

    print(
        f"  Both:     seq={result['t_both_seq']:.3f} ms, "
        f"DEER={result['t_both_deer']:.3f} ms, "
        f"speedup={result['both_speedup']:.3f}x"
    )

    print(
        f"  Errors: forward={result['forward_error']:.3e}, "
        f"backward={result['backward_error']:.3e}"
    )

    print(
        f"  Newton steps: forward={result['fwd_newton_steps']}, "
        f"backward={result['bwd_newton_steps']}"
    )

    print()


# ============================================================
# Summary table
# ============================================================

print("\n================ Summary Table ================\n")

header = (
    f"{'n':>5} {'m':>5} {'T':>6} | "
    f"{'Fwd Seq':>10} {'Fwd DEER':>10} {'Fwd Spd':>9} | "
    f"{'Bwd Seq':>10} {'Bwd DEER':>10} {'Bwd Spd':>9} | "
    f"{'Both Seq':>10} {'Both DEER':>10} {'Both Spd':>9}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} {r['T']:6d} | "
        f"{r['t_fwd_seq']:10.3f} {r['t_fwd_deer']:10.3f} {r['forward_speedup']:9.3f} | "
        f"{r['t_bwd_seq']:10.3f} {r['t_bwd_deer']:10.3f} {r['backward_speedup']:9.3f} | "
        f"{r['t_both_seq']:10.3f} {r['t_both_deer']:10.3f} {r['both_speedup']:9.3f}"
    )


# ============================================================
# Optional: print accuracy summary
# ============================================================

print("\n================ Accuracy Summary ================\n")

for r in results:
    print(
        f"n={r['n']:3d}, m={r['m']:3d}: "
        f"forward error={r['forward_error']:.3e}, "
        f"backward error={r['backward_error']:.3e}, "
        f"grad norm={r['grad_norm']:.3e}"
    )