"""
run_pendulum_two_pass_deer_nn.py

Execution script for Pendulum-v1 with a neural-network policy using
DEER in a decoupled two-pass form:

    Pass 1: forward DEER for states
        x_{k+1} = F(x_k, theta)

    Pass 2: backward DEER for costates
        lambda_k = l_x(x_k, theta) + F_x(x_k, theta)^T lambda_{k+1}

Then compute the deterministic policy gradient:

    grad_theta J = sum_k grad_theta [l(x_k, theta)
                    + lambda_{k+1}^T F(x_k, theta)]

Requires:
    deer.py with deer_alg
    pendulum_env_jax.py
"""

import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

from deer import deer_alg

import pendulum_env_jax as penv


# ============================================================
# Configuration
# ============================================================

SEED = 0
T_HORIZON = 200          # Gymnasium Pendulum-v1 truncates at 200 steps
NUM_MC_SAMPLES = 4       # keep small first
NUM_POLICY_ITERS = 30

DEER_MAX_ITERS = 15
DEER_TOL = 1e-8

LEARNING_RATE = 1e-4
GRAD_CLIP_NORM = 50.0

HIDDEN_DIM = 16


# ============================================================
# Small PyTree helpers
# ============================================================

def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a, b):
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_scale(a, scalar):
    return jax.tree_util.tree_map(lambda x: scalar * x, a)


def tree_mean(trees):
    n = len(trees)
    total = trees[0]
    for tree in trees[1:]:
        total = tree_add(total, tree)
    return tree_scale(total, 1.0 / n)


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))


def clip_tree_by_norm(tree, max_norm):
    norm = tree_l2_norm(tree)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return tree_scale(tree, scale), norm


# ============================================================
# Robust DEER result parser
# ============================================================

def parse_deer_result(result):
    trajectory = result[1]
    steps = result[2] if len(result) > 2 else None
    return trajectory, steps


# ============================================================
# Two-pass decoupled DEER for one initial state
# ============================================================

def two_pass_deer_gradient_single(x0, params, key):
    """
    Compute policy gradient for one initial state using:
        1. forward DEER state solve
        2. backward DEER costate solve
        3. deterministic policy-gradient formula
    """
    key_x, key_lam = jr.split(key)

    # ------------------------------------------------------------
    # Pass 1: forward DEER for state trajectory
    # ------------------------------------------------------------

    def forward_f(x, dummy):
        return penv.pendulum_step(x, params)

    # Simple initial guess: repeat x0 with small noise.
    states_guess = jnp.tile(x0, (T_HORIZON, 1)) + 1e-2 * jr.normal(
        key_x,
        shape=(T_HORIZON, penv.STATE_DIM),
    )

    dummy_inputs = jnp.zeros((T_HORIZON, 1))

    forward_result = deer_alg(
        forward_f,
        x0,
        states_guess,
        dummy_inputs,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )

    states_deer, fwd_steps = parse_deer_result(forward_result)

    # states_deer = [x_1, ..., x_T]
    # x_traj      = [x_0, ..., x_{T-1}]
    x_traj = jnp.vstack([x0, states_deer[:-1]])

    # ------------------------------------------------------------
    # Pass 2: backward DEER for costate trajectory
    # ------------------------------------------------------------

    lambda_T = jnp.zeros(penv.STATE_DIM)

    def grad_stage_x(x):
        return jax.grad(lambda y: penv.stage_cost(y, params))(x)

    def F_x(x):
        return jax.jacrev(lambda y: penv.pendulum_step(y, params))(x)

    def backward_f(lambda_next, x_k):
        return grad_stage_x(x_k) + F_x(x_k).T @ lambda_next

    costate_guess = 1e-2 * jr.normal(
        key_lam,
        shape=(T_HORIZON, penv.STATE_DIM),
    )

    x_traj_rev = jnp.flip(x_traj, axis=0)

    backward_result = deer_alg(
        backward_f,
        lambda_T,
        costate_guess,
        x_traj_rev,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )

    lambda_reversed, bwd_steps = parse_deer_result(backward_result)

    # lambda_reversed      = [lambda_{T-1}, ..., lambda_0]
    # lambda_chronological = [lambda_0, ..., lambda_{T-1}]
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)

    lambda_k_plus_1 = jnp.vstack([
        lambda_chronological[1:],
        lambda_T[None, :],
    ])

    # ------------------------------------------------------------
    # Policy gradient from Hamiltonian
    # ------------------------------------------------------------

    def grad_step(x_k, lambda_next):
        def h(p):
            return (
                penv.stage_cost(x_k, p)
                + jnp.dot(lambda_next, penv.pendulum_step(x_k, p))
            )
        return jax.grad(h)(params)

    grad_terms = jax.vmap(grad_step)(x_traj, lambda_k_plus_1)

    # Sum over time for each parameter leaf.
    gradient = jax.tree_util.tree_map(lambda g: jnp.sum(g, axis=0), grad_terms)

    # For monitoring only.
    costs = jax.vmap(lambda x: penv.stage_cost(x, params))(x_traj)
    total_cost = jnp.sum(costs)

    return gradient, total_cost, fwd_steps, bwd_steps


# ============================================================
# Monte Carlo gradient
# ============================================================

def monte_carlo_deer_gradient(params, x0_samples, key):
    sample_count = int(x0_samples.shape[0])
    keys = jr.split(key, sample_count)

    gradients = []
    costs = []
    fwd_steps = []
    bwd_steps = []

    for i in range(sample_count):
        grad_i, cost_i, fwd_i, bwd_i = two_pass_deer_gradient_single(
            x0_samples[i],
            params,
            keys[i],
        )
        gradients.append(grad_i)
        costs.append(cost_i)

        if fwd_i is not None:
            fwd_steps.append(float(np.asarray(fwd_i)))
        if bwd_i is not None:
            bwd_steps.append(float(np.asarray(bwd_i)))

    mean_grad = tree_mean(gradients)
    mean_cost = float(jnp.mean(jnp.asarray(costs)))

    mean_fwd_steps = float(np.mean(fwd_steps)) if fwd_steps else np.nan
    mean_bwd_steps = float(np.mean(bwd_steps)) if bwd_steps else np.nan

    return mean_grad, mean_cost, mean_fwd_steps, mean_bwd_steps


# ============================================================
# Main optimization loop
# ============================================================

def main():
    master_key = jr.PRNGKey(SEED)
    key_params, key_samples, key_train = jr.split(master_key, 3)

    params = penv.init_policy_params(key_params, hidden_dim=HIDDEN_DIM, scale=0.1)
    x0_samples = penv.sample_initial_states(key_samples, NUM_MC_SAMPLES)

    history_cost = []
    history_grad_norm = []

    print("Two-pass decoupled DEER on Pendulum-v1 with NN policy")
    print(f"T_HORIZON={T_HORIZON}, NUM_MC_SAMPLES={NUM_MC_SAMPLES}")
    print(f"DEER_MAX_ITERS={DEER_MAX_ITERS}, LEARNING_RATE={LEARNING_RATE}\n")

    start = time.perf_counter()

    for iteration in range(NUM_POLICY_ITERS):
        iter_key = jr.fold_in(key_train, iteration)

        grad, mean_cost, mean_fwd_steps, mean_bwd_steps = monte_carlo_deer_gradient(
            params,
            x0_samples,
            iter_key,
        )

        grad, grad_norm = clip_tree_by_norm(grad, GRAD_CLIP_NORM)

        # Gradient descent on cost. Gym reward is negative cost.
        params = tree_sub(params, tree_scale(grad, LEARNING_RATE))

        history_cost.append(mean_cost)
        history_grad_norm.append(float(grad_norm))

        print(
            f"iter={iteration:03d} | "
            f"cost={mean_cost:10.4f} | "
            f"grad_norm={float(grad_norm):10.4e} | "
            f"fwd_steps={mean_fwd_steps:.1f} | "
            f"bwd_steps={mean_bwd_steps:.1f}"
        )

    elapsed = time.perf_counter() - start
    print(f"\nTotal training time: {elapsed:.3f} seconds")

    # Plot cost history.
    plt.figure(figsize=(7, 5))
    plt.plot(history_cost, marker="o", markersize=3)
    plt.xlabel("Policy-gradient iteration")
    plt.ylabel("Mean finite-horizon cost")
    plt.title("Pendulum NN policy trained with two-pass DEER gradient")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("pendulum_two_pass_deer_cost.png", dpi=200)
    plt.show()

    # Plot one final trajectory from a fixed test state.
    x0_test = jnp.array([jnp.pi - 0.2, 0.0])
    x_traj, x_T = penv.rollout_sequential(x0_test, params, T_HORIZON)
    x_full = jnp.vstack([x_traj, x_T[None, :]])

    theta = jax.vmap(penv.angle_normalize)(x_full[:, 0])
    theta_dot = x_full[:, 1]

    plt.figure(figsize=(7, 5))
    plt.plot(np.asarray(theta), label="theta normalized")
    plt.plot(np.asarray(theta_dot), label="theta_dot")
    plt.xlabel("Time step")
    plt.ylabel("State")
    plt.title("Final policy rollout")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pendulum_two_pass_deer_final_rollout.png", dpi=200)
    plt.show()

    np.savez(
        "pendulum_two_pass_deer_nn_results.npz",
        cost=np.asarray(history_cost),
        grad_norm=np.asarray(history_grad_norm),
        theta=np.asarray(theta),
        theta_dot=np.asarray(theta_dot),
    )

    print("Saved:")
    print("  pendulum_two_pass_deer_cost.png")
    print("  pendulum_two_pass_deer_final_rollout.png")
    print("  pendulum_two_pass_deer_nn_results.npz")


if __name__ == "__main__":
    main()
