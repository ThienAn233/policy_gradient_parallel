"""
pendulum_env_jax.py

Differentiable JAX version of the Gymnasium Pendulum-v1 dynamics.

State used here:
    x = [theta, theta_dot]

Observation used by the neural-network policy:
    obs = [cos(theta), sin(theta), theta_dot]

Cost is the negative of the Gymnasium reward:
    cost = theta_normalized^2 + 0.1 * theta_dot^2 + 0.001 * torque^2
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr


STATE_DIM = 2
OBS_DIM = 3
ACTION_DIM = 1

G = 10.0
M = 1.0
L = 1.0
DT = 0.05
MAX_SPEED = 8.0
MAX_TORQUE = 2.0


def angle_normalize(theta):
    """Normalize angle to [-pi, pi]."""
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def state_to_obs(x):
    """
    Convert internal state [theta, theta_dot] to Gymnasium-style observation:
        [cos(theta), sin(theta), theta_dot]
    """
    theta, theta_dot = x
    return jnp.array([jnp.cos(theta), jnp.sin(theta), theta_dot])


def init_policy_params(key, hidden_dim=16, scale=0.1):
    """
    Small MLP policy:
        obs -> tanh hidden -> tanh hidden -> scalar torque

    The output torque is bounded later by MAX_TORQUE * tanh(raw_action).
    """
    k1, k2, k3 = jr.split(key, 3)

    W1 = scale * jr.normal(k1, shape=(OBS_DIM, hidden_dim))
    b1 = jnp.zeros(hidden_dim)

    W2 = scale * jr.normal(k2, shape=(hidden_dim, hidden_dim))
    b2 = jnp.zeros(hidden_dim)

    W3 = scale * jr.normal(k3, shape=(hidden_dim, ACTION_DIM))
    b3 = jnp.zeros(ACTION_DIM)

    return (W1, b1, W2, b2, W3, b3)


def policy_raw(params, obs):
    W1, b1, W2, b2, W3, b3 = params

    h1 = jnp.tanh(obs @ W1 + b1)
    h2 = jnp.tanh(h1 @ W2 + b2)
    raw = h2 @ W3 + b3

    return raw[0]


def policy(params, x):
    """Return bounded torque in [-MAX_TORQUE, MAX_TORQUE]."""
    obs = state_to_obs(x)
    return MAX_TORQUE * jnp.tanh(policy_raw(params, obs))


def stage_cost(x, params):
    """
    Gymnasium Pendulum cost, i.e. negative reward.

    reward = -(theta^2 + 0.1 theta_dot^2 + 0.001 torque^2)
    """
    theta, theta_dot = x
    torque = policy(params, x)
    theta_norm = angle_normalize(theta)

    return theta_norm**2 + 0.1 * theta_dot**2 + 0.001 * torque**2


def pendulum_step(x, params):
    """
    One Gymnasium-style Pendulum-v1 dynamics step.

    This follows the Gymnasium implementation:
        new_theta_dot = theta_dot + (3g/(2l) sin(theta)
                        + 3/(m l^2) torque) dt
        new_theta_dot = clip(new_theta_dot, -8, 8)
        new_theta = theta + new_theta_dot dt
    """
    theta, theta_dot = x
    torque = policy(params, x)

    new_theta_dot = theta_dot + (
        3.0 * G / (2.0 * L) * jnp.sin(theta)
        + 3.0 / (M * L**2) * torque
    ) * DT

    new_theta_dot = jnp.clip(new_theta_dot, -MAX_SPEED, MAX_SPEED)
    new_theta = theta + new_theta_dot * DT

    return jnp.array([new_theta, new_theta_dot])


def sample_initial_states(key, sample_count):
    """
    Gymnasium Pendulum-v1 reset distribution:
        theta     ~ Uniform[-pi, pi]
        theta_dot ~ Uniform[-1, 1]
    """
    k1, k2 = jr.split(key)

    theta = jr.uniform(
        k1,
        shape=(sample_count,),
        minval=-jnp.pi,
        maxval=jnp.pi,
    )

    theta_dot = jr.uniform(
        k2,
        shape=(sample_count,),
        minval=-1.0,
        maxval=1.0,
    )

    return jnp.stack([theta, theta_dot], axis=1)


def rollout_sequential(x0, params, horizon):
    """
    Sequential rollout, used only for monitoring/plotting.

    Returns:
        x_traj = [x_0, ..., x_{T-1}], shape (T, 2)
        x_T
    """
    def step(x, _):
        x_next = pendulum_step(x, params)
        return x_next, x

    x_T, x_traj = jax.lax.scan(step, x0, xs=None, length=horizon)
    return x_traj, x_T


def trajectory_cost(x0, params, horizon):
    """Total finite-horizon cost using sequential rollout."""
    x_traj, _ = rollout_sequential(x0, params, horizon)
    costs = jax.vmap(lambda x: stage_cost(x, params))(x_traj)
    return jnp.sum(costs)
