"""
run_pendulum_two_pass_deer_nn_batched_terminal.py

Two-pass decoupled DEER policy-gradient optimization for Pendulum
with a neural-network policy.

This version includes:
    - batch dimension over initial states using vmap
    - terminal costate lambda_T = nabla_x J_T(x_T)
    - final trajectory plots
    - pendulum snapshot visualization
    - optional GIF animation

Requires:
    deer.py
    pendulum_env_jax.py
"""

import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter

from deer import deer_alg

from pendulum_env_jax import (
    STATE_DIM,
    ACTION_DIM,
    MAX_TORQUE,
    init_mlp_params,
    mlp_policy,
    closed_loop_step,
    stage_cost,
    lambda_terminal,
)


# ============================================================
# 1. Settings
# ============================================================

SEED = 0

T_HORIZON = 200
BATCH_SIZE = 16
NUM_POLICY_ITERS = 50

DEER_MAX_ITERS = 5
DEER_TOL = 1e-7

LEARNING_RATE = 1e-3
TERMINAL_WEIGHT = 1.0

RESULT_DIR = Path("pendulum_two_pass_deer_nn_terminal_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Pytree helpers
# ============================================================

def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_mul_scalar(a, scalar):
    return jax.tree_util.tree_map(lambda x: scalar * x, a)


def tree_mean(a, axis=0):
    return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=axis), a)


def tree_sum(a, axis=0):
    return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=axis), a)


def tree_norm(a):
    leaves = jax.tree_util.tree_leaves(a)
    return jnp.sqrt(sum([jnp.sum(x**2) for x in leaves]))


# ============================================================
# 3. Sampling initial states
# ============================================================

def sample_initial_states(key, batch_size):
    """
    Sample initial states:
        theta     ~ Uniform(-pi, pi)
        theta_dot ~ Uniform(-1, 1)
    """
    k1, k2 = jr.split(key)

    theta = jr.uniform(
        k1,
        shape=(batch_size,),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )

    theta_dot = jr.uniform(
        k2,
        shape=(batch_size,),
        minval=-1.0,
        maxval=1.0,
    )

    return jnp.stack([theta, theta_dot], axis=1)


# ============================================================
# 4. Sequential rollout for visualization only
# ============================================================

def rollout_policy(x0, params):
    def step(x, _):
        u = mlp_policy(params, x)
        x_next = closed_loop_step(x, params)
        c = stage_cost(x, params)
        return x_next, (x, u, c)

    x_T, (xs, us, costs) = jax.lax.scan(
        step,
        x0,
        xs=None,
        length=T_HORIZON,
    )

    xs_full = jnp.vstack([xs, x_T[None, :]])
    return xs_full, us, costs


# ============================================================
# 5. Two-pass DEER gradient for one initial state
# ============================================================

def two_pass_deer_gradient_single(x0, params, key):
    """
    One-sample two-pass DEER policy gradient.

    Forward pass:
        x_{k+1} = F(x_k, params)

    Backward pass:
        lambda_k = l_x(x_k, params)
                   + F_x(x_k, params)^T lambda_{k+1}

    Terminal:
        lambda_T = nabla_x J_T(x_T)

    Policy gradient:
        grad_params J = sum_k grad_params[
            l(x_k, params) + lambda_{k+1}^T F(x_k, params)
        ]
    """
    key_x, key_lam = jr.split(key)

    dummy_inputs = jnp.zeros((T_HORIZON, ACTION_DIM))

    # ------------------------------------------------------------
    # Forward DEER
    # ------------------------------------------------------------

    def forward_f(x, dummy):
        return closed_loop_step(x, params)

    x_guess = 0.01 * jr.normal(
        key_x,
        shape=(T_HORIZON, STATE_DIM),
    )

    # Use full_trace=True so this is fixed-iteration and vmap-friendly.
    forward_result = deer_alg(
        forward_f,
        x0,
        x_guess,
        dummy_inputs,
        num_iters=DEER_MAX_ITERS,
        full_trace=True,
        Ts=None,
        tol=DEER_TOL,
    )

    x_states = forward_result[1]          # [x_1, ..., x_T]
    x_T = x_states[-1]

    # x_traj = [x_0, ..., x_{T-1}]
    x_traj = jnp.vstack([x0, x_states[:-1]])

    # Terminal costate lambda_T = nabla_x J_T(x_T)
    lambda_T = lambda_terminal(x_T, TERMINAL_WEIGHT)

    # ------------------------------------------------------------
    # Backward DEER
    # ------------------------------------------------------------

    def backward_f(lambda_next, x_k):
        l_x = jax.grad(lambda x: stage_cost(x, params))(x_k)

        F_x = jax.jacrev(lambda x: closed_loop_step(x, params))(x_k)

        return l_x + F_x.T @ lambda_next

    lambda_guess = 0.01 * jr.normal(
        key_lam,
        shape=(T_HORIZON, STATE_DIM),
    )

    x_traj_rev = jnp.flip(x_traj, axis=0)

    backward_result = deer_alg(
        backward_f,
        lambda_T,
        lambda_guess,
        x_traj_rev,
        num_iters=DEER_MAX_ITERS,
        full_trace=True,
        Ts=None,
        tol=DEER_TOL,
    )

    lambda_reversed = backward_result[1]  # [lambda_{T-1}, ..., lambda_0]

    # Convert to [lambda_0, ..., lambda_{T-1}]
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)

    lambda_k_plus_1 = jnp.vstack([
        lambda_chronological[1:],
        lambda_T[None, :],
    ])

    # ------------------------------------------------------------
    # Policy-gradient terms
    # ------------------------------------------------------------

    def hamiltonian_param_grad(x_k, lambda_next):
        def H(p):
            return (
                stage_cost(x_k, p)
                + jnp.dot(lambda_next, closed_loop_step(x_k, p))
            )

        return jax.grad(H)(params)

    grad_terms = jax.vmap(
        hamiltonian_param_grad,
        in_axes=(0, 0),
    )(x_traj, lambda_k_plus_1)

    grad_params = tree_sum(grad_terms, axis=0)

    total_cost = (
        jnp.sum(jax.vmap(lambda x: stage_cost(x, params))(x_traj))
        + TERMINAL_WEIGHT * 0.0
    )

    return grad_params, total_cost


# ============================================================
# 6. Batch with vmap
# ============================================================

batched_two_pass_deer_gradient = jax.jit(
    jax.vmap(
        two_pass_deer_gradient_single,
        in_axes=(0, None, 0),
        out_axes=(0, 0),
    )
)


def monte_carlo_gradient(params, x0_batch, key):
    keys = jr.split(key, x0_batch.shape[0])

    grad_batch, cost_batch = batched_two_pass_deer_gradient(
        x0_batch,
        params,
        keys,
    )

    mean_grad = tree_mean(grad_batch, axis=0)
    mean_cost = jnp.mean(cost_batch)

    return mean_grad, mean_cost


# ============================================================
# 7. Policy optimization
# ============================================================

key = jr.PRNGKey(SEED)

key_params, key_samples, key_train = jr.split(key, 3)

params = init_mlp_params(key_params, hidden_dim=32)

x0_batch = sample_initial_states(key_samples, BATCH_SIZE)

history_cost = []
history_grad_norm = []

print("Starting Pendulum NN policy optimization with two-pass DEER")
print(f"T_HORIZON={T_HORIZON}, BATCH_SIZE={BATCH_SIZE}, ITERS={NUM_POLICY_ITERS}")
print(f"Terminal costate uses lambda_T = grad_x J_T(x_T), TERMINAL_WEIGHT={TERMINAL_WEIGHT}")

start = time.perf_counter()

for it in range(NUM_POLICY_ITERS):
    iter_key = jr.fold_in(key_train, it)

    grad_params, mean_cost = monte_carlo_gradient(
        params,
        x0_batch,
        iter_key,
    )

    grad_norm = tree_norm(grad_params)

    params = tree_sub(
        params,
        tree_mul_scalar(grad_params, LEARNING_RATE),
    )

    history_cost.append(float(mean_cost))
    history_grad_norm.append(float(grad_norm))

    if it == 0 or (it + 1) % 5 == 0 or it + 1 == NUM_POLICY_ITERS:
        print(
            f"iter={it:03d} | "
            f"cost={float(mean_cost):.6f} | "
            f"grad_norm={float(grad_norm):.6e}"
        )

elapsed = time.perf_counter() - start

print(f"\nTraining time: {elapsed:.3f} seconds")


# ============================================================
# 8. Final rollout and visualization
# ============================================================

x0_eval = jnp.array([jnp.pi - 0.1, 0.0])
xs_final, us_final, costs_final = rollout_policy(x0_eval, params)

xs_np = np.asarray(xs_final)
us_np = np.asarray(us_final)
costs_np = np.asarray(costs_final)

theta = xs_np[:, 0]
theta_dot = xs_np[:, 1]
time_axis = np.arange(xs_np.shape[0])

# Save numerical results
np.savez(
    RESULT_DIR / "pendulum_two_pass_deer_terminal_results.npz",
    theta=theta,
    theta_dot=theta_dot,
    controls=us_np,
    costs=costs_np,
    history_cost=np.asarray(history_cost),
    history_grad_norm=np.asarray(history_grad_norm),
)

# ------------------------------------------------------------
# Plot cost history
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))
plt.plot(history_cost, marker="o", markersize=3)
plt.xlabel("Policy iteration")
plt.ylabel("Mean trajectory cost")
plt.title("Training cost")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULT_DIR / "training_cost.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# Plot final trajectory
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.plot(time_axis, theta)
plt.ylabel(r"$\theta$")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time_axis, theta_dot)
plt.ylabel(r"$\dot{\theta}$")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(np.arange(len(us_np)), us_np)
plt.ylabel("u")
plt.xlabel("time step")
plt.grid(True)

plt.suptitle("Final learned-policy pendulum trajectory")
plt.tight_layout()
plt.savefig(RESULT_DIR / "final_trajectory.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# Pendulum snapshots through time
# ------------------------------------------------------------

snapshot_indices = np.linspace(0, len(theta) - 1, 10, dtype=int)

plt.figure(figsize=(10, 3))
for idx, t_idx in enumerate(snapshot_indices):
    th = theta[t_idx]

    # Pendulum position. Upright is theta=0.
    px = np.sin(th)
    py = np.cos(th)

    ax = plt.subplot(1, len(snapshot_indices), idx + 1)
    ax.plot([0, px], [0, py], linewidth=2)
    ax.scatter([px], [py], s=40)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"t={t_idx}")

plt.suptitle("Pendulum snapshots through time")
plt.tight_layout()
plt.savefig(RESULT_DIR / "pendulum_snapshots.png", dpi=200)
plt.show()

# ------------------------------------------------------------
# Optional GIF animation
# ------------------------------------------------------------

try:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.grid(True)

    line, = ax.plot([], [], linewidth=3)
    bob, = ax.plot([], [], "o", markersize=12)

    def init():
        line.set_data([], [])
        bob.set_data([], [])
        return line, bob

    def update(frame):
        th = theta[frame]
        px = np.sin(th)
        py = np.cos(th)

        line.set_data([0, px], [0, py])
        bob.set_data([px], [py])
        ax.set_title(f"t = {frame}")
        return line, bob

    anim = FuncAnimation(
        fig,
        update,
        frames=len(theta),
        init_func=init,
        interval=40,
        blit=True,
    )

    anim.save(
        RESULT_DIR / "pendulum_final_animation.gif",
        writer=PillowWriter(fps=25),
    )

    plt.close(fig)

except Exception as e:
    print("GIF animation was skipped:", e)

print(f"\nSaved results to: {RESULT_DIR.resolve()}")