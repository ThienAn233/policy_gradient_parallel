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
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def time_jax(name, fn, warmup=1, repeat=5):
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

    print(f"{name:<45}: {mean_ms:10.3f} ms ± {std_ms:8.3f} ms")
    return out, mean_ms, std_ms


# ============================================================
# System Definition
# ============================================================

T_max = 300
D = 3

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
K = jr.normal(key, shape=(D, D)) * 0.1

x0 = jnp.ones(D)
lambda_T = jnp.zeros(D)

tol = 1e-7
deer_iters = T_max
warmup = 1
repeat = 5

F_cl = A - B @ K
F_cl_T = F_cl.T
x_ref = jnp.ones(D)


# ============================================================
# Forward dynamics
# ============================================================

def f(x, u_dummy):
    return F_cl @ x


def rollout_step(x, _):
    x_next = f(x, None)
    return x_next, x_next


@jax.jit
def manual_forward_rollout():
    _, states_rollout = jax.lax.scan(
        rollout_step,
        x0,
        jnp.arange(T_max),
    )
    return states_rollout


# ============================================================
# Costate dynamics
# ============================================================

def backward_costate_step(lambda_next, x_k):
    """
    lambda_k = grad_x l(x_k, K) + F_x^T lambda_{k+1}

    l(x,u) = ||x - 1||^2 + ||u||^2
    u = -Kx
    """
    u_k = -K @ x_k

    grad_x_l = 2.0 * (x_k - x_ref) - 2.0 * K.T @ u_k

    lambda_k = grad_x_l + F_cl_T @ lambda_next

    return lambda_k


def back_step(lambda_next, x_k):
    lambda_k = backward_costate_step(lambda_next, x_k)
    return lambda_k, lambda_k


@jax.jit
def manual_backward_costate(states_rollout):
    """
    Returns costates in backward-stacked order:

        [lambda_{T-1}, ..., lambda_0]
    """
    x_traj = jnp.vstack([x0, states_rollout[:-1]])

    _, lambda_traj_rev = jax.lax.scan(
        back_step,
        lambda_T,
        jnp.flip(x_traj, axis=0),
    )

    return lambda_traj_rev


@jax.jit
def manual_forward_backward():
    states_rollout = manual_forward_rollout()
    lambda_traj_rev = manual_backward_costate(states_rollout)
    return states_rollout, lambda_traj_rev


# ============================================================
# DEER guesses
# ============================================================

states_guess = jr.normal(
    jr.PRNGKey(1),
    shape=(T_max, D),
)

costate_guess = jr.normal(
    jr.PRNGKey(2),
    shape=(T_max, D),
)

dummy_inputs = jnp.zeros((T_max, D))


# ============================================================
# Two-pass decoupled DEER
# ============================================================

@jax.jit
def deer_forward_pass():
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


@jax.jit
def deer_backward_pass(states_driver):
    """
    Backward DEER pass with states fixed as drivers.

    states_driver is [x_1, ..., x_T].
    """
    x_traj = jnp.vstack([x0, states_driver[:-1]])

    x_traj_rev = jnp.flip(x_traj, axis=0)

    _, costate_deer, newton_steps, *_ = deer_alg_fixed_j(
        backward_costate_step,
        F_cl_T,
        lambda_T,
        costate_guess,
        x_traj_rev,
        num_iters=deer_iters,
        full_trace=False,
        Ts=None,
        tol=tol,
    )

    return costate_deer, newton_steps


@jax.jit
def deer_two_pass_decoupled():
    states_deer, fwd_steps = deer_forward_pass()
    costate_deer, bwd_steps = deer_backward_pass(states_deer)
    return states_deer, costate_deer, fwd_steps, bwd_steps


# ============================================================
# Option 1: One-pass decoupled augmented DEER
# ============================================================

def f_augmented(z_k, x_rev_driver_k):
    """
    One-pass decoupled augmented dynamics.

    z_k = [x_k, mu_k],
    where mu_k = lambda_{T-k}.

    The state part uses x_k from z_k.

    The costate part uses a fixed reversed state driver:
        x_rev_driver_k = x_{T-1-k}.

    This intentionally omits G_x Delta x coupling.
    """
    x_k = z_k[:D]
    mu_k = z_k[D:]

    x_next = F_cl @ x_k

    # mu_next = lambda_{T-k-1}
    mu_next = backward_costate_step(mu_k, x_rev_driver_k)

    return jnp.concatenate([x_next, mu_next])


def make_one_pass_driver(states_driver):
    """
    Build driver sequence for the one-pass augmented method.

    states_driver is [x_1, ..., x_T].

    x_traj is [x_0, ..., x_{T-1}].

    x_rev_driver is [x_{T-1}, ..., x_0].
    """
    x_traj = jnp.vstack([x0, states_driver[:-1]])
    x_rev_driver = jnp.flip(x_traj, axis=0)
    return x_rev_driver


def make_z_guess():
    """
    z_guess is [z_1, ..., z_T].

    z_k = [x_k, mu_k],
    mu_k = lambda_{T-k}.

    Therefore:
        z_guess[:, :D] = [x_1, ..., x_T]
        z_guess[:, D:] = [lambda_{T-1}, ..., lambda_0]
    """
    return jnp.concatenate([states_guess, costate_guess], axis=1)


z0 = jnp.concatenate([x0, lambda_T])
z_guess = make_z_guess()


@jax.jit
def deer_one_pass_decoupled_fixed_driver(states_driver):
    """
    One DEER call on z = [x, reversed lambda].

    The costate uses states_driver as a fixed driver.
    """
    x_rev_driver = make_one_pass_driver(states_driver)

    _, z_deer, newton_steps, *_ = deer_alg_fixed_j(
        f_augmented,
        jnp.block([
            [F_cl, jnp.zeros((D, D))],
            [jnp.zeros((D, D)), F_cl_T],
        ]),
        z0,
        z_guess,
        x_rev_driver,
        num_iters=deer_iters,
        full_trace=False,
        Ts=None,
        tol=tol,
    )

    states_one_pass = z_deer[:, :D]
    costates_one_pass_rev = z_deer[:, D:]

    return states_one_pass, costates_one_pass_rev, newton_steps


# ============================================================
# Timing
# ============================================================

print("\n================ Timing Results ================\n")

baseline_out, t_seq, _ = time_jax(
    "Manual sequential forward + backward",
    manual_forward_backward,
    warmup=warmup,
    repeat=repeat,
)

states_rollout, lambda_manual_rev = baseline_out

two_pass_out, t_two_pass, _ = time_jax(
    "DEER decoupled two-pass",
    deer_two_pass_decoupled,
    warmup=warmup,
    repeat=repeat,
)

states_two_pass, lambda_two_pass_rev, fwd_steps, bwd_steps = two_pass_out

# For Option 1, use manual rollout as fixed driver so the costate target is consistent.
one_pass_out, t_one_pass, _ = time_jax(
    "DEER decoupled one-pass fixed driver",
    lambda: deer_one_pass_decoupled_fixed_driver(states_rollout),
    warmup=warmup,
    repeat=repeat,
)

states_one_pass, lambda_one_pass_rev, one_pass_steps = one_pass_out


# ============================================================
# Accuracy checks
# ============================================================

print("\n================ Accuracy Checks ================\n")

two_pass_state_error = jnp.max(jnp.abs(states_two_pass - states_rollout))
two_pass_costate_error = jnp.max(jnp.abs(lambda_two_pass_rev - lambda_manual_rev))

one_pass_state_error = jnp.max(jnp.abs(states_one_pass - states_rollout))
one_pass_costate_error = jnp.max(jnp.abs(lambda_one_pass_rev - lambda_manual_rev))

one_vs_two_state_error = jnp.max(jnp.abs(states_one_pass - states_two_pass))
one_vs_two_costate_error = jnp.max(jnp.abs(lambda_one_pass_rev - lambda_two_pass_rev))

print("Two-pass state error      :", two_pass_state_error)
print("Two-pass costate error    :", two_pass_costate_error)
print("One-pass state error      :", one_pass_state_error)
print("One-pass costate error    :", one_pass_costate_error)
print("One-pass vs two-pass state error   :", one_vs_two_state_error)
print("One-pass vs two-pass costate error :", one_vs_two_costate_error)

print("\nTwo-pass forward Newton steps :", fwd_steps)
print("Two-pass backward Newton steps:", bwd_steps)
print("One-pass Newton steps         :", one_pass_steps)


# ============================================================
# Speed comparison
# ============================================================

print("\n================ Speed Summary ================\n")

print(f"Sequential time : {t_seq:10.3f} ms")
print(f"Two-pass DEER   : {t_two_pass:10.3f} ms")
print(f"One-pass DEER   : {t_one_pass:10.3f} ms")

print(f"\nTwo-pass speedup over sequential: {t_seq / t_two_pass:8.3f}x")
print(f"One-pass speedup over sequential: {t_seq / t_one_pass:8.3f}x")
print(f"One-pass speedup over two-pass  : {t_two_pass / t_one_pass:8.3f}x")


# ============================================================
# Policy gradient from manual baseline
# ============================================================

lambda_manual = jnp.flip(lambda_manual_rev, axis=0)

x_traj = jnp.vstack([x0, states_rollout[:-1]])

lambda_k_plus_1_traj = jnp.vstack([
    lambda_manual[1:],
    lambda_T,
])


def compute_grad_step(carry, inputs):
    x_k, lambda_k_plus_1 = inputs

    u_k = -K @ x_k

    h_u = 2.0 * u_k + B.T @ lambda_k_plus_1

    grad_K_step = -jnp.outer(h_u, x_k)

    return carry, grad_K_step


_, all_K_grads = jax.lax.scan(
    compute_grad_step,
    None,
    (x_traj, lambda_k_plus_1_traj),
)

grad_K = jnp.sum(all_K_grads, axis=0)

print("\nGradient w.r.t K:\n", grad_K)

learning_rate = 1e-4
K_new = K - learning_rate * grad_K

print("\nUpdated Feedback Matrix K:\n", K_new)