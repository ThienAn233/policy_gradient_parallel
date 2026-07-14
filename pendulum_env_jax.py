"""
pendulum_env_jax.py

Small differentiable JAX version of the Gymnasium Pendulum dynamics.

State:
    x = [theta, theta_dot]

Policy input/observation:
    obs = [cos(theta), sin(theta), theta_dot]

Action:
    u in [-2, 2]
"""

import jax
import jax.numpy as jnp


# ============================================================
# Pendulum constants
# ============================================================

G = 10.0
M = 1.0
L = 1.0
DT = 0.05

MAX_SPEED = 8.0
MAX_TORQUE = 2.0

STATE_DIM = 2
OBS_DIM = 3
ACTION_DIM = 1


# ============================================================
# Basic helpers
# ============================================================

def angle_normalize(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def state_to_obs(x):
    theta = x[0]
    theta_dot = x[1]
    return jnp.array([
        jnp.cos(theta),
        jnp.sin(theta),
        theta_dot,
    ])


# ============================================================
# Neural-network policy
# ============================================================

def init_mlp_params(key, hidden_dim=32):
    """
    Simple MLP:
        obs(3) -> hidden -> hidden -> action(1)

    Returns a pytree/list of (W,b).
    """
    k1, k2, k3 = jax.random.split(key, 3)

    W1 = 0.1 * jax.random.normal(k1, (hidden_dim, OBS_DIM))
    b1 = jnp.zeros(hidden_dim)

    W2 = 0.1 * jax.random.normal(k2, (hidden_dim, hidden_dim))
    b2 = jnp.zeros(hidden_dim)

    W3 = 0.1 * jax.random.normal(k3, (ACTION_DIM, hidden_dim))
    b3 = jnp.zeros(ACTION_DIM)

    return [(W1, b1), (W2, b2), (W3, b3)]


def mlp_policy(params, x):
    """
    Deterministic bounded policy:
        u = 2 tanh(MLP(obs))
    """
    obs = state_to_obs(x)

    h = obs
    for W, b in params[:-1]:
        h = jnp.tanh(W @ h + b)

    W_last, b_last = params[-1]
    raw_u = W_last @ h + b_last

    u = MAX_TORQUE * jnp.tanh(raw_u[0])
    return u


# ============================================================
# Dynamics and costs
# ============================================================

def pendulum_step_state(x, u):
    """
    Gymnasium-style Pendulum dynamics.

    State:
        x = [theta, theta_dot]
    """
    theta = x[0]
    theta_dot = x[1]

    u = jnp.clip(u, -MAX_TORQUE, MAX_TORQUE)

    new_theta_dot = theta_dot + (
        3.0 * G / (2.0 * L) * jnp.sin(theta)
        + 3.0 / (M * L**2) * u
    ) * DT

    new_theta_dot = jnp.clip(new_theta_dot, -MAX_SPEED, MAX_SPEED)

    new_theta = theta + new_theta_dot * DT

    return jnp.array([new_theta, new_theta_dot])


def closed_loop_step(x, params):
    u = mlp_policy(params, x)
    return pendulum_step_state(x, u)


def stage_cost(x, params):
    """
    Positive cost corresponding to negative Gymnasium reward.

    cost = theta_normalized^2 + 0.1 theta_dot^2 + 0.001 u^2
    """
    theta = x[0]
    theta_dot = x[1]
    u = mlp_policy(params, x)

    theta_error = angle_normalize(theta)

    return theta_error**2 + 0.1 * theta_dot**2 + 0.001 * u**2


def terminal_cost(x_T, terminal_weight=1.0):
    """
    Terminal cost J_T(x_T).

    This is independent of the policy parameters.
    """
    theta = x_T[0]
    theta_dot = x_T[1]

    theta_error = angle_normalize(theta)

    return terminal_weight * (
        theta_error**2 + 0.1 * theta_dot**2
    )


def lambda_terminal(x_T, terminal_weight=1.0):
    """
    Terminal costate:
        lambda_T = nabla_x J_T(x_T)
    """
    return jax.grad(
        lambda x: terminal_cost(x, terminal_weight)
    )(x_T)