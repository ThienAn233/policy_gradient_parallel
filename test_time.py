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
#
# NOTE:
# The one-pass method uses z = [x; lambda], so deer_alg forms a
# (2n) x (2n) Jacobian. This can become very memory-heavy for large n.
# Start with the safe configs, then uncomment larger sizes if your machine
# has enough memory.
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

    lambda_T = jnp.zeros(n)
    x_ref = jnp.ones(n)

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
        u_k = -K @ x_k

        # grad_x ||x - x_ref||^2 + grad_x ||-Kx||^2
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

    def both_sequential_raw():
        states_rollout = forward_sequential_raw()
        lambda_traj_rev = backward_sequential_given_states_raw(states_rollout)
        return states_rollout, lambda_traj_rev

    both_sequential = jax.jit(both_sequential_raw)

    # ------------------------------------------------------------
    # Decoupled two-pass DEER
    # ------------------------------------------------------------

    def deer_forward_pass_raw():
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

    deer_forward_pass = jax.jit(deer_forward_pass_raw)

    def deer_backward_pass_raw(states_driver):
        """
        Backward DEER pass with state trajectory fixed as driver.

        states_driver is [x_1, ..., x_T].
        """
        x_traj = jnp.vstack([x0, states_driver[:-1]])
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

    deer_backward_pass = jax.jit(deer_backward_pass_raw)

    def deer_two_pass_raw():
        states_deer, fwd_steps = deer_forward_pass_raw()
        costate_deer, bwd_steps = deer_backward_pass_raw(states_deer)
        return states_deer, costate_deer, fwd_steps, bwd_steps

    deer_two_pass = jax.jit(deer_two_pass_raw)

    # ------------------------------------------------------------
    # Option 1: Decoupled one-pass augmented DEER
    # ------------------------------------------------------------

    def make_one_pass_driver(states_driver):
        """
        Build fixed reversed state driver for the one-pass augmented method.

        states_driver is [x_1, ..., x_T].
        x_traj       is [x_0, ..., x_{T-1}].
        x_rev_driver is [x_{T-1}, ..., x_0].
        """
        x_traj = jnp.vstack([x0, states_driver[:-1]])
        x_rev_driver = jnp.flip(x_traj, axis=0)
        return x_rev_driver

    def f_augmented(z_k, x_rev_driver_k):
        """
        Decoupled one-pass augmented dynamics.

        z_k = [x_k, mu_k], where mu_k = lambda_{T-k}.

        State:
            x_{k+1} = F(x_k)

        Reversed costate:
            mu_{k+1} = lambda_{T-k-1}
                     = G(x_{T-k-1}, mu_k)

        The driver x_{T-k-1} is fixed during this DEER solve.
        This omits the coupled Newton term G_x Delta x.
        """
        x_k = z_k[:n]
        mu_k = z_k[n:]

        x_next = f(x_k, None)
        mu_next = backward_costate_step(mu_k, x_rev_driver_k)

        return jnp.concatenate([x_next, mu_next])

    z0 = jnp.concatenate([x0, lambda_T])
    z_guess = jnp.concatenate([states_guess, costate_guess], axis=1)

    def deer_one_pass_raw(states_driver_for_costate):
        """
        One DEER call on z = [x, reversed lambda].

        The costate part uses states_driver_for_costate as fixed driver.
        For a clean comparison, pass the manual rollout.
        """
        x_rev_driver = make_one_pass_driver(states_driver_for_costate)

        _, z_deer, newton_steps, *_ = deer_alg(
            f_augmented,
            z0,
            z_guess,
            x_rev_driver,
            num_iters=deer_iters,
            full_trace=False,
            Ts=None,
            tol=tol,
        )

        states_one_pass = z_deer[:, :n]
        costates_one_pass_rev = z_deer[:, n:]

        return states_one_pass, costates_one_pass_rev, newton_steps

    deer_one_pass = jax.jit(deer_one_pass_raw)

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

    # Option 1 uses the baseline state rollout as fixed costate driver.
    one_out, t_one, sd_one = time_jax(
        lambda: deer_one_pass(states_rollout),
        warmup=warmup,
        repeat=repeat,
    )
    states_one, lambda_one_rev, one_steps = one_out

    # ------------------------------------------------------------
    # Accuracy checks against manual sequential baseline
    # ------------------------------------------------------------

    two_state_error = float(jnp.max(jnp.abs(states_two - states_rollout)))
    two_costate_error = float(jnp.max(jnp.abs(lambda_two_rev - lambda_seq_rev)))

    one_state_error = float(jnp.max(jnp.abs(states_one - states_rollout)))
    one_costate_error = float(jnp.max(jnp.abs(lambda_one_rev - lambda_seq_rev)))

    one_vs_two_state_error = float(jnp.max(jnp.abs(states_one - states_two)))
    one_vs_two_costate_error = float(jnp.max(jnp.abs(lambda_one_rev - lambda_two_rev)))

    # ------------------------------------------------------------
    # Gradient norm from sequential baseline
    # ------------------------------------------------------------

    lambda_traj = jnp.flip(lambda_seq_rev, axis=0)
    x_traj = jnp.vstack([x0, states_rollout[:-1]])
    lambda_k_plus_1_traj = jnp.vstack([lambda_traj[1:], lambda_T])

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

        "t_one": t_one,
        "sd_one": sd_one,

        "two_speedup": t_seq / t_two,
        "one_speedup": t_seq / t_one,
        "one_vs_two_speedup": t_two / t_one,

        "two_state_error": two_state_error,
        "two_costate_error": two_costate_error,

        "one_state_error": one_state_error,
        "one_costate_error": one_costate_error,

        "one_vs_two_state_error": one_vs_two_state_error,
        "one_vs_two_costate_error": one_vs_two_costate_error,

        "fwd_newton_steps": int(fwd_steps),
        "bwd_newton_steps": int(bwd_steps),
        "one_newton_steps": int(one_steps),

        "grad_norm": grad_norm,
    }


# ============================================================
# Run all benchmarks
# ============================================================

results = []

print("\n================ Shape Scaling Benchmark ================\n")
print("Methods:")
print("  1. Sequential baseline")
print("  2. Decoupled two-pass DEER")
print("  3. Decoupled one-pass DEER with fixed costate driver")
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
        f"  One-pass:   {fmt_time(result['t_one'], result['sd_one'])} ms, "
        f"speedup={result['one_speedup']:.3f}x, "
        f"vs two-pass={result['one_vs_two_speedup']:.3f}x"
    )

    print(
        f"  Errors two-pass: state={result['two_state_error']:.3e}, "
        f"costate={result['two_costate_error']:.3e}"
    )

    print(
        f"  Errors one-pass: state={result['one_state_error']:.3e}, "
        f"costate={result['one_costate_error']:.3e}"
    )

    print(
        f"  Newton steps: two-pass=({result['fwd_newton_steps']}, "
        f"{result['bwd_newton_steps']}), "
        f"one-pass={result['one_newton_steps']}"
    )

    print()


# ============================================================
# Summary table: timing with standard deviation
# ============================================================

print("\n================ Timing Summary Table ================\n")

header = (
    f"{'n':>5} {'m':>5} {'T':>6} | "
    f"{'Seq mean±std (ms)':>22} | "
    f"{'Two mean±std (ms)':>22} {'Two Spd':>9} | "
    f"{'One mean±std (ms)':>22} {'One Spd':>9} {'One/Two':>9}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} {r['T']:6d} | "
        f"{fmt_time(r['t_seq'], r['sd_seq']):>22} | "
        f"{fmt_time(r['t_two'], r['sd_two']):>22} {r['two_speedup']:9.3f} | "
        f"{fmt_time(r['t_one'], r['sd_one']):>22} {r['one_speedup']:9.3f} "
        f"{r['one_vs_two_speedup']:9.3f}"
    )


# ============================================================
# Summary table: accuracy
# ============================================================

print("\n================ Accuracy Summary Table ================\n")

header = (
    f"{'n':>5} {'m':>5} | "
    f"{'Two x err':>12} {'Two lam err':>12} | "
    f"{'One x err':>12} {'One lam err':>12} | "
    f"{'One-vs-Two x':>14} {'One-vs-Two lam':>16}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} | "
        f"{r['two_state_error']:12.3e} {r['two_costate_error']:12.3e} | "
        f"{r['one_state_error']:12.3e} {r['one_costate_error']:12.3e} | "
        f"{r['one_vs_two_state_error']:14.3e} "
        f"{r['one_vs_two_costate_error']:16.3e}"
    )


# ============================================================
# Summary table: Newton iterations
# ============================================================

print("\n================ Newton Step Summary ================\n")

header = (
    f"{'n':>5} {'m':>5} | "
    f"{'Two fwd':>8} {'Two bwd':>8} {'One':>8} | "
    f"{'grad norm':>12}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['n']:5d} {r['m']:5d} | "
        f"{r['fwd_newton_steps']:8d} {r['bwd_newton_steps']:8d} "
        f"{r['one_newton_steps']:8d} | "
        f"{r['grad_norm']:12.3e}"
    )