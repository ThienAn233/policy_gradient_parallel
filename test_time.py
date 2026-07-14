import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
from deer_LQR import deer_alg_fixed_j


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


def time_jax(fn, warmup=1, repeat=10):
    """
    Time a JAX function correctly.

    The warmup calls compile the function, so they are not counted.
    Returns:
        out, mean_ms, std_ms
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


def fmt_time(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


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


def make_system(n, m, T_max, seed=0, radius=0.85):
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
    k1, k2, k3, k4, k5 = jr.split(key, 5)

    F_cl = make_stable_matrix(k1, n, radius=radius)

    B = jr.normal(k2, shape=(n, m)) / jnp.sqrt(max(m, 1))
    K = 0.1 * jr.normal(k3, shape=(m, n)) / jnp.sqrt(max(n, 1))

    A = F_cl + B @ K

    x0 = jnp.ones(n)

    states_guess = jr.normal(k4, shape=(T_max, n))
    costate_guess = jr.normal(k5, shape=(T_max, n))

    dummy_inputs = jnp.zeros((T_max, m))

    return A, B, K, x0, states_guess, costate_guess, dummy_inputs


# ============================================================
# Global experiment settings
# ============================================================

T_max = 1000
tol = 1e-7
repeat = 10
warmup = 1
deer_iters = 50

# Each pair is (state dimension n, control dimension m).
configs = [
    (3, 1),
    (16, 8),
    (32, 16),
    (64, 32),
    (128, 64),
    (256, 128),
    (512, 256),
    # (1024, 512),
]


# ============================================================
# One benchmark for one (n, m)
# ============================================================

def run_one_benchmark(n, m, seed=0):
    A, B, K, x0, states_guess, costate_guess, dummy_inputs = make_system(
        n=n,
        m=m,
        T_max=T_max,
        seed=seed,
        radius=0.85,
    )

    x_ref = jnp.zeros(n)

    F_cl = A - B @ K
    F_x_T = F_cl.T

    # ------------------------------------------------------------
    # Closed-loop dynamics
    # ------------------------------------------------------------

    def f(x, u_dummy):
        """
        Closed-loop dynamics:
            u = -K x
            x_next = A x + B u = (A - B K) x
        """
        u = -K @ x
        return A @ x + B @ u

    def grad_ell_x(x):
        """
        Gradient of the stage cost ell(x, K).

        ell(x, K) = ||x - x_ref||^2 + ||u||^2, u = -Kx

        This is also used as the terminal costate:
            lambda_T = nabla_x ell(x_T, K)
        """
        u = -K @ x
        return 2.0 * (x - x_ref) - 2.0 * K.T @ u

    # ------------------------------------------------------------
    # Manual sequential forward rollout
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
    # Costate dynamics
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
        grad_x_l = grad_ell_x(x_k)

        lambda_k = grad_x_l + F_x_T @ lambda_next

        return lambda_k

    def back_step(lambda_next, x_k):
        lambda_k = backward_costate_step(lambda_next, x_k)
        return lambda_k, lambda_k

    def backward_sequential_given_states_raw(states_rollout):
        """
        Returns costates in reverse order:
            [lambda_{T-1}, ..., lambda_0]

        Terminal condition:
            lambda_T = nabla_x ell(x_T, K)
        """
        x_T = states_rollout[-1]
        lambda_T = grad_ell_x(x_T)

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

    def both_sequential_raw():
        states_rollout = forward_sequential_raw()
        lambda_traj_rev = backward_sequential_given_states_raw(states_rollout)
        return states_rollout, lambda_traj_rev

    both_sequential = jax.jit(both_sequential_raw)

    # ------------------------------------------------------------
    # Decoupled two-pass DEER
    # ------------------------------------------------------------

    def deer_forward_pass_raw():
        _, states_deer, newton_steps, *_ = deer_alg_fixed_j(
            f,
            F_cl,
            x0,
            states_guess,
            dummy_inputs,
            num_iters=deer_iters,
            full_trace=False,
            Ts=None,
            tol=tol,
        )
        return states_deer, newton_steps

    deer_forward_pass = jax.jit(deer_forward_pass_raw)

    def deer_backward_pass_raw(states_driver):
        """
        Backward DEER pass with state trajectory fixed as driver.

        states_driver is [x_1, ..., x_T].

        Terminal condition:
            lambda_T = nabla_x ell(x_T, K)
        """
        x_T = states_driver[-1]
        lambda_T = grad_ell_x(x_T)

        x_traj = jnp.vstack([x0, states_driver[:-1]])
        x_traj_rev = jnp.flip(x_traj, axis=0)

        _, costate_deer, newton_steps, *_ = deer_alg_fixed_j(
            backward_costate_step,
            F_x_T,
            lambda_T,
            costate_guess,
            x_traj_rev,
            num_iters=deer_iters,
            full_trace=False,
            Ts=None,
            tol=tol,
        )

        return costate_deer, newton_steps

    deer_backward_pass = jax.jit(deer_backward_pass_raw)

    def deer_two_pass_raw():
        states_deer, fwd_steps = deer_forward_pass_raw()
        costate_deer, bwd_steps = deer_backward_pass_raw(states_deer)
        return states_deer, costate_deer, fwd_steps, bwd_steps

    deer_two_pass = jax.jit(deer_two_pass_raw)

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------

    seq_out, t_seq, sd_seq = time_jax(
        both_sequential,
        warmup=warmup,
        repeat=repeat,
    )
    states_rollout, lambda_seq_rev = seq_out

    two_out, t_two, sd_two = time_jax(
        deer_two_pass,
        warmup=warmup,
        repeat=repeat,
    )
    states_two, lambda_two_rev, fwd_steps, bwd_steps = two_out

    # ------------------------------------------------------------
    # Accuracy checks against manual sequential baseline
    # ------------------------------------------------------------

    two_state_error = float(jnp.max(jnp.abs(states_two - states_rollout)))
    two_costate_error = float(jnp.max(jnp.abs(lambda_two_rev - lambda_seq_rev)))

    # ------------------------------------------------------------
    # Gradient norm from sequential baseline
    # ------------------------------------------------------------

    lambda_traj = jnp.flip(lambda_seq_rev, axis=0)
    x_traj = jnp.vstack([x0, states_rollout[:-1]])

    # lambda_T = nabla_x ell(x_T, K)
    lambda_T = grad_ell_x(states_rollout[-1])

    lambda_k_plus_1_traj = jnp.vstack([
        lambda_traj[1:],
        lambda_T[None, :],
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

        "t_seq": t_seq,
        "sd_seq": sd_seq,

        "t_two": t_two,
        "sd_two": sd_two,

        "two_speedup": t_seq / t_two,
        "two_state_error": two_state_error,
        "two_costate_error": two_costate_error,

        "fwd_newton_steps": int(fwd_steps),
        "bwd_newton_steps": int(bwd_steps),
        "grad_norm": grad_norm,
    }


# ============================================================
# Run all benchmarks
# ============================================================

results = []

print("\n================ Two-Pass DEER Shape Scaling Benchmark ================\n")
print("Methods:")
print("  1. Sequential baseline")
print("  2. Decoupled two-pass DEER")
print()

for idx, (n, m) in enumerate(configs):
    print(f"Running benchmark for n={n}, m={m} ...")

    result = run_one_benchmark(
        n=n,
        m=m,
        seed=idx,
    )

    results.append(result)

    print(f"  Sequential: {fmt_time(result['t_seq'], result['sd_seq'])} ms")

    print(
        f"  Two-pass:   {fmt_time(result['t_two'], result['sd_two'])} ms, "
        f"speedup={result['two_speedup']:.3f}x"
    )

    print(
        f"  Errors two-pass: state={result['two_state_error']:.3e}, "
        f"costate={result['two_costate_error']:.3e}"
    )

    print(
        f"  Newton steps: two-pass=({result['fwd_newton_steps']}, "
        f"{result['bwd_newton_steps']})"
    )

    print()


# ============================================================
# Summary table: timing with standard deviation
# ============================================================

print("\n================ Timing Summary Table ================\n")

header = (
    f"{'n':>5} {'m':>5} {'T':>6} | "
    f"{'Seq mean±std (ms)':>22} | "
    f"{'Two mean±std (ms)':>22} {'Two Spd':>9}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} {r['T']:6d} | "
        f"{fmt_time(r['t_seq'], r['sd_seq']):>22} | "
        f"{fmt_time(r['t_two'], r['sd_two']):>22} {r['two_speedup']:9.3f}"
    )


# ============================================================
# Summary table: accuracy
# ============================================================

print("\n================ Accuracy Summary Table ================\n")

header = (
    f"{'n':>5} {'m':>5} | "
    f"{'Two x err':>12} {'Two lam err':>12}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} | "
        f"{r['two_state_error']:12.3e} {r['two_costate_error']:12.3e}"
    )


# ============================================================
# Summary table: Newton iterations
# ============================================================

print("\n================ Newton Step Summary ================\n")

header = (
    f"{'n':>5} {'m':>5} | "
    f"{'Two fwd':>8} {'Two bwd':>8} | "
    f"{'grad norm':>12}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} | "
        f"{r['fwd_newton_steps']:8d} {r['bwd_newton_steps']:8d} "
        f"{r['grad_norm']:12.3e}"
    )