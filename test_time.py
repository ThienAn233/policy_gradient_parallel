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


def time_jax(name, fn, warmup=1, repeat=5):
    """
    Time a JAX function correctly.

    warmup: first calls compile the function, so do not count them.
    repeat: number of actual timed runs.
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

    print(f"{name:<42}: {mean_ms:10.3f} ms  ± {std_ms:8.3f} ms")
    return out, mean_ms, std_ms


# ============================================================
# System Definition
# ============================================================

T_max = 300

A = jnp.array([
    [-1.0, 0.0,  0.0],
    [ 0.0, 0.7,  0.0],
    [ 0.0, 0.0, -0.2],
])

B = jnp.array([
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
])

key = jr.PRNGKey(0)

K = jax.random.normal(key, shape=(3, 3)) * 0.1
x0 = jnp.array([1.0, 1.0, 1.0])

tol = 1e-7
repeat = 5
warmup = 1


# ============================================================
# Closed-loop dynamics
# ============================================================

def f(x, u_dummy):
    """
    Closed-loop dynamics:

        x_{k+1} = (A - B K) x_k
    """
    return (A - B @ K) @ x


# ============================================================
# Sequential forward rollout
# ============================================================

def rollout_step(x, _):
    x_next = f(x, None)
    return x_next, x_next


def forward_sequential():
    _, states_rollout = jax.lax.scan(
        rollout_step,
        x0,
        jnp.arange(T_max),
    )
    return states_rollout


# ============================================================
# Parallel / DEER forward rollout
# ============================================================

states_guess = jax.random.normal(
    jr.PRNGKey(1),
    shape=(T_max, 3),
)

dummy_inputs = jnp.zeros((T_max, 3))


def forward_deer():
    """
    DEER forward solve using associative_scan inside deer_alg.

    Use full_trace=False for timing final trajectory only.
    full_trace=True stores all iterates and is slower.
    """
    _, states_deer, newton_steps, *_ = deer_alg(
        f,
        x0,
        states_guess,
        dummy_inputs,
        num_iters=T_max,
        full_trace=False,
        Ts=None,
        tol=tol,
    )
    return states_deer, newton_steps


# ============================================================
# Backward costate dynamics
# ============================================================

def backward_costate_step(lambda_next, x_k):
    """
    Computes

        lambda_k = grad_x l(x_k, K) + F_x^T lambda_{k+1}.
    """
    u_k = -K @ x_k

    # Example cost:
    #     l(x, u) = ||x - 1||^2 + ||u||^2
    #
    # Since u = -Kx,
    #     grad_x ||u||^2 = -2 K^T u.
    grad_x_l = 2 * (x_k - 1.0) - 2 * K.T @ u_k

    F_x_T = (A - B @ K).T

    lambda_k = grad_x_l + F_x_T @ lambda_next
    return lambda_k


def back_step(lambda_next, x_k):
    lambda_k = backward_costate_step(lambda_next, x_k)
    return lambda_k, lambda_k


lambda_T = jnp.zeros(3)

costate_guess = jax.random.normal(
    jr.PRNGKey(2),
    shape=(T_max, 3),
)


def backward_sequential_given_states(states_rollout):
    """
    Sequential backward costate pass.

    Returns costates in reversed order:

        [lambda_{T-1}, ..., lambda_0].
    """
    x_traj = jnp.vstack([x0, states_rollout[:-1]])

    _, lambda_traj_rev = jax.lax.scan(
        back_step,
        lambda_T,
        jnp.flip(x_traj, axis=0),
    )

    return lambda_traj_rev


def backward_deer_given_states(states_rollout):
    """
    DEER backward solve.

    Returns costates in reversed order:

        [lambda_{T-1}, ..., lambda_0].
    """
    x_traj = jnp.vstack([x0, states_rollout[:-1]])
    x_traj_rev = jnp.flip(x_traj, axis=0)

    _, costate_deer, newton_steps, *_ = deer_alg(
        backward_costate_step,
        lambda_T,
        costate_guess,
        x_traj_rev,
        num_iters=T_max,
        full_trace=False,
        Ts=None,
        tol=tol,
    )

    return costate_deer, newton_steps


# ============================================================
# Combined forward + backward functions
# ============================================================

def both_sequential():
    states_rollout = forward_sequential()
    lambda_traj_rev = backward_sequential_given_states(states_rollout)
    return states_rollout, lambda_traj_rev


def both_deer():
    states_deer, fwd_steps = forward_deer()
    costate_deer, bwd_steps = backward_deer_given_states(states_deer)
    return states_deer, costate_deer, fwd_steps, bwd_steps


# ============================================================
# Run timing comparison
# ============================================================

print("\n================ Timing Results ================\n")

states_rollout, t_fwd_seq, _ = time_jax(
    "Sequential forward rollout",
    forward_sequential,
    warmup=warmup,
    repeat=repeat,
)

forward_deer_out, t_fwd_deer, _ = time_jax(
    "DEER / parallel forward rollout",
    forward_deer,
    warmup=warmup,
    repeat=repeat,
)

states_deer, fwd_newton_steps = forward_deer_out

lambda_seq_rev, t_bwd_seq, _ = time_jax(
    "Sequential backward costate",
    lambda: backward_sequential_given_states(states_rollout),
    warmup=warmup,
    repeat=repeat,
)

backward_deer_out, t_bwd_deer, _ = time_jax(
    "DEER / parallel backward costate",
    lambda: backward_deer_given_states(states_rollout),
    warmup=warmup,
    repeat=repeat,
)

costate_deer, bwd_newton_steps = backward_deer_out

both_seq_out, t_both_seq, _ = time_jax(
    "Sequential forward + backward",
    both_sequential,
    warmup=warmup,
    repeat=repeat,
)

both_deer_out, t_both_deer, _ = time_jax(
    "DEER / parallel forward + backward",
    both_deer,
    warmup=warmup,
    repeat=repeat,
)

states_seq_both, lambda_seq_both_rev = both_seq_out
states_deer_both, lambda_deer_both_rev, fwd_steps_both, bwd_steps_both = both_deer_out


# ============================================================
# Accuracy checks
# ============================================================

print("\n================ Accuracy Checks ================\n")

forward_error = jnp.max(jnp.abs(states_deer - states_rollout))
backward_error = jnp.max(jnp.abs(costate_deer - lambda_seq_rev))

print("Forward Pass Error  (DEER vs sequential):", forward_error)
print("Backward Pass Error (DEER vs sequential):", backward_error)

both_forward_error = jnp.max(jnp.abs(states_deer_both - states_seq_both))
both_backward_error = jnp.max(jnp.abs(lambda_deer_both_rev - lambda_seq_both_rev))

print("Both Forward Error  (DEER vs sequential):", both_forward_error)
print("Both Backward Error (DEER vs sequential):", both_backward_error)

print("\nForward DEER Newton steps:", fwd_newton_steps)
print("Backward DEER Newton steps:", bwd_newton_steps)


# ============================================================
# Speedup summary
# ============================================================

print("\n================ Speedup Summary ================\n")

print(f"Forward speedup:  {t_fwd_seq / t_fwd_deer:8.3f}x")
print(f"Backward speedup: {t_bwd_seq / t_bwd_deer:8.3f}x")
print(f"Both speedup:     {t_both_seq / t_both_deer:8.3f}x")


# ============================================================
# Compute Policy Gradient using sequential costates
# ============================================================

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
    h_u = 2 * u_k + B.T @ lambda_k_plus_1

    # Since u = -Kx:
    #     grad_K u = -x
    # so:
    #     grad_K J step = - h_u x_k^T
    grad_K_step = -jnp.outer(h_u, x_k)

    return carry, grad_K_step


_, all_K_grads = jax.lax.scan(
    compute_grad_step,
    None,
    (x_traj, lambda_k_plus_1_traj),
)

grad_K = jnp.sum(all_K_grads, axis=0)

print("\nGradient w.r.t K:\n", grad_K)


# ============================================================
# Gradient update
# ============================================================

learning_rate = 1e-4
K_new = K - learning_rate * grad_K

print("\nUpdated Feedback Matrix K:\n", K_new)