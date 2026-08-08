"""
Monte Carlo projected policy-gradient optimization for the inertia-wheel
pendulum.

The implementation follows a decoupled two-pass DEER structure:

    Pass 1: forward DEER for the closed-loop state trajectory
        x_{k+1} = F(x_k, theta)

    Pass 2: backward DEER for the costate trajectory
        lambda_k = l_x(x_k, theta)
                   + F_x(x_k, theta).T @ lambda_{k+1}

The policy parameters are

    theta = [a_1, a_2, a_3, k_p, k_v].

For each Monte Carlo initial state, the code evaluates

    grad_theta J = sum_k (grad_theta pi(x_k, theta)).T
                   @ (2 R pi(x_k, theta) + g.T @ lambda_{k+1}).

The gradients are averaged across initial-state samples. Each unconstrained
batch-gradient step is then projected onto the feasible policy-parameter set:

    theta_trial <- theta - learning_rate * mean_gradient
    theta       <- argmin_z 0.5 * ||z - theta_trial||_2^2
                   subject to h_i(z) >= feasibility_margin.

Thus the constraints are enforced by projected gradient descent (PGD), not by
adding a log-barrier term to the objective. No Adam or momentum is used.

Required files/packages:
    - deer.py containing deer_alg, available on the Python path
    - pip install "jax[cpu]" numpy scipy matplotlib pillow
"""

from __future__ import annotations
from collections.abc import Callable, Sequence
from typing import Optional

import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from scipy.optimize import minimize

# Use the DEER implementation supplied with your project.
from deer import deer_alg


# =============================================================================
# Configuration
# =============================================================================
SEED = 0

DT = 0.01
T_HORIZON = 1000
NUM_MC_SAMPLES = 128
NUM_POLICY_ITERS = 80000

DEER_MAX_ITERS = 5000
DEER_TOL = 1.0e-8

# The gradient formula is a sum over the full horizon, so begin with a small
# learning rate. This is projected gradient descent, not Adam.
LEARNING_RATE = 1.0e-5
FEASIBILITY_MARGIN = 1.0e-8
PROJECTION_FTOL = 1.0e-12
PROJECTION_MAX_ITERS = 200

PRINT_EVERY = 5
RESULTS_DIR = Path("inertia_wheel_projected_gradient_two_pass_deer_mc_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIM = 4
PARAM_DIM = 5

ConditionFunction = Callable[[jax.Array], jax.Array]
ConditionGradientFunction = Callable[[jax.Array], jax.Array]
# =============================================================================
# Inertia-wheel pendulum model and IDA-PBC policy
# =============================================================================
m_11 = 0.1
m_22 = 0.2
m_3 = 10.0

# x = [q_1, q_2, p_1, p_2]
# Continuous-time input direction: x_dot = f_c(x) + g_c u.
g_c = jnp.array([0.0, 0.0, -1.0, 1.0])

# For forward Euler, x_{k+1} = f(x_k) + g pi(x_k, theta).
g = DT * g_c

# Running cost l(x, theta) = x^T Q x + R pi(x, theta)^2.
# DT makes the finite sum approximate a continuous-time integral.
Q = DT * jnp.diag(jnp.array([12.0, 15, 0.20, 0.08]))
R = DT * 2.0
Q_f = Q

# Initial controller values from the paper example.
theta_initial = jnp.array([1.0, -1.5, 6.0, 3.75, 10.0])

# ---------------------------------------------------------
# Condition functions h_i(theta) > 0
# ---------------------------------------------------------

def condition_a1_positive(theta: jax.Array) -> jax.Array:
    """Condition h_1(theta) = a_1 > 0."""
    a_1 = theta[0]
    return a_1


def condition_md_determinant(theta: jax.Array) -> jax.Array:
    """
    Condition h_2(theta) = det(M_d) > 0, where

        M_d = [[a_1, a_2],
               [a_2, a_3]].
    """
    a_1, a_2, a_3 = theta[:3]
    return a_1 * a_3 - a_2**2


def condition_a1_plus_a2_negative(
    theta: jax.Array,
) -> jax.Array:
    """
    The paper requires a_1 + a_2 < 0.

    It is written in the common h(theta) >= 0 form as

        h_3(theta) = -(a_1 + a_2) > 0.
    """
    a_1, a_2 = theta[:2]
    return -(a_1 + a_2)


def condition_kp_positive(theta: jax.Array) -> jax.Array:
    """Condition h_4(theta) = k_p > 0."""
    k_p = theta[3]
    return k_p


def condition_kv_positive(theta: jax.Array) -> jax.Array:
    """Condition h_5(theta) = k_v > 0."""
    k_v = theta[4]
    return k_v


condition_functions: list[ConditionFunction] = [
    condition_a1_positive,
    condition_md_determinant,
    condition_a1_plus_a2_negative,
    condition_kp_positive,
    condition_kv_positive,
]

# ---------------------------------------------------------
# Explicit gradients of the condition functions
# ---------------------------------------------------------

def gradient_a1_positive(theta: jax.Array) -> jax.Array:
    """Gradient of h_1(theta) = a_1."""
    return jnp.array(
        [1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=theta.dtype,
    )


def gradient_md_determinant(theta: jax.Array) -> jax.Array:
    """
    Gradient of

        h_2(theta) = a_1 a_3 - a_2^2.
    """
    a_1, a_2, a_3 = theta[:3]

    return jnp.array(
        [
            a_3,
            -2.0 * a_2,
            a_1,
            0.0,
            0.0,
        ],
        dtype=theta.dtype,
    )


def gradient_a1_plus_a2_negative(
    theta: jax.Array,
) -> jax.Array:
    """
    Gradient of

        h_3(theta) = -(a_1 + a_2).
    """
    return jnp.array(
        [-1.0, -1.0, 0.0, 0.0, 0.0],
        dtype=theta.dtype,
    )


def gradient_kp_positive(theta: jax.Array) -> jax.Array:
    """Gradient of h_4(theta) = k_p."""
    return jnp.array(
        [0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=theta.dtype,
    )


def gradient_kv_positive(theta: jax.Array) -> jax.Array:
    """Gradient of h_5(theta) = k_v."""
    return jnp.array(
        [0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=theta.dtype,
    )


condition_gradient_functions: list[ConditionGradientFunction] = [
    gradient_a1_positive,
    gradient_md_determinant,
    gradient_a1_plus_a2_negative,
    gradient_kp_positive,
    gradient_kv_positive,
]

def controller_constants(theta: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute gamma_1, gamma_2, and k_2 from [a_1,a_2,a_3,k_p,k_v]."""
    a_1, a_2, a_3, _, _ = theta

    gamma_1 = (a_2 / (a_1 + a_2)) * m_3
    gamma_2 = -m_11 * (a_2 + a_3) / (m_22 * (a_1 + a_2))
    k_2 = -m_22 * (a_1 + a_2) / (a_1 * a_3 - a_2**2)
    return gamma_1, gamma_2, k_2


def pi(x: jax.Array, theta: jax.Array) -> jax.Array:
    """IDA-PBC control law u = pi(x, theta)."""
    q_1, q_2, p_1, p_2 = x
    _, _, _, k_p, k_v = theta
    gamma_1, gamma_2, k_2 = controller_constants(theta)

    q_1_dot = p_1 / m_11
    q_2_dot = p_2 / m_22

    return (
        gamma_1 * jnp.sin(q_1)
        + k_p * (q_2 + gamma_2 * q_1)
        + k_v * k_2 * (q_2_dot + gamma_2 * q_1_dot)
    )


def f(x: jax.Array) -> jax.Array:
    """Unforced forward-Euler map f(x) = x + DT f_c(x)."""
    q_1, q_2, p_1, p_2 = x
    x_dot_unforced = jnp.array(
        [
            p_1 / m_11,
            p_2 / m_22,
            m_3 * jnp.sin(q_1),
            0.0,
        ]
    )
    return x + DT * x_dot_unforced


def closed_loop_step(x: jax.Array, theta: jax.Array) -> jax.Array:
    """Closed-loop system F(x,theta) = f(x) + g pi(x,theta)."""
    return f(x) + g * pi(x, theta)


def stage_cost(x: jax.Array, theta: jax.Array) -> jax.Array:
    u = pi(x, theta)
    return x @ Q @ x + R * u**2


def terminal_cost(x: jax.Array) -> jax.Array:
    return x @ Q_f @ x


# =============================================================================
# Monte Carlo initial-state distribution
# =============================================================================
def sample_initial_states(key: jax.Array, sample_count: int) -> jax.Array:
    """Sample initial conditions around the downward configuration q_1 = pi.

    The velocity samples are converted to momenta using
        p_1 = m_11 q_1_dot,
        p_2 = m_22 q_2_dot.
    Change these intervals to match the region in which the controller should
    perform well.
    """
    key_q1, key_q2, key_dq1, key_dq2 = jr.split(key, 4)

    q_1 = jr.uniform(
        key_q1,
        shape=(sample_count,),
        minval=jnp.pi - 0.35,
        maxval=jnp.pi + 0.35,
    )
    q_2 = jr.uniform(
        key_q2,
        shape=(sample_count,),
        minval=-0.25,
        maxval=0.25,
    )
    q_1_dot = jr.uniform(
        key_dq1,
        shape=(sample_count,),
        minval=-0.50,
        maxval=0.50,
    )
    q_2_dot = jr.uniform(
        key_dq2,
        shape=(sample_count,),
        minval=-1.00,
        maxval=1.00,
    )

    p_1 = m_11 * q_1_dot
    p_2 = m_22 * q_2_dot
    return jnp.stack((q_1, q_2, p_1, p_2), axis=1)


# =============================================================================
# DEER helpers
# =============================================================================
def parse_deer_result(result):
    """Read the trajectory and iteration count returned by deer_alg."""
    trajectory = result[1]
    steps = result[2] if len(result) > 2 else jnp.asarray(-1)
    return trajectory, steps


def forward_deer_single(
    x_0: jax.Array,
    theta: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pass 1: solve x_{k+1}=F(x_k,theta) with the supplied deer_alg."""

    def forward_f(x_k, _dummy):
        return closed_loop_step(x_k, theta)

    # Guess for [x_1,...,x_T]. Small noise avoids identical initial guesses.
    states_guess = jnp.broadcast_to(x_0, (T_HORIZON, STATE_DIM))
    states_guess = states_guess + 0.05 * jr.normal(
        key,
        shape=(T_HORIZON, STATE_DIM),
    )
    dummy_inputs = jnp.zeros((T_HORIZON, 1))

    result = deer_alg(
        forward_f,
        x_0,
        states_guess,
        dummy_inputs,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )
    states_tail, steps = parse_deer_result(result)

    # states_tail = [x_1,...,x_T].
    states_full = jnp.vstack((x_0, states_tail))
    return states_full, steps


grad_stage_x = jax.grad(stage_cost, argnums=0)
jac_closed_loop_x = jax.jacrev(closed_loop_step, argnums=0)
grad_terminal_x = jax.grad(terminal_cost)


def backward_deer_single(
    states_full: jax.Array,
    theta: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pass 2: solve the backward costate recurrence with deer_alg."""

    def backward_f(lambda_next, x_k):
        return (
            grad_stage_x(x_k, theta)
            + jac_closed_loop_x(x_k, theta).T @ lambda_next
        )

    # x_traj contains x_0,...,x_{T-1}. DEER advances through reversed time.
    x_traj = states_full[:-1]
    x_traj_reversed = jnp.flip(x_traj, axis=0)

    # lambda_T = grad_x terminal_cost(x_T).
    lambda_T = grad_terminal_x(states_full[-1])

    costate_guess = 0.05 * jr.normal(
        key,
        shape=(T_HORIZON, STATE_DIM),
    )

    result = deer_alg(
        backward_f,
        lambda_T,
        costate_guess,
        x_traj_reversed,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )
    lambda_reversed, steps = parse_deer_result(result)

    # lambda_reversed = [lambda_{T-1},...,lambda_0].
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)
    costates_full = jnp.vstack((lambda_chronological, lambda_T))
    return costates_full, steps


# =============================================================================
# Nonlinear policy gradient
# =============================================================================
grad_pi_theta = jax.jacrev(pi, argnums=1)


def policy_gradient(
    states_full: jax.Array,
    costates_full: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    """Evaluate the requested state-costate policy-gradient formula."""
    x_k = states_full[:-1]
    lambda_k_plus_1 = costates_full[1:]

    u_k = jax.vmap(pi, in_axes=(0, None))(x_k, theta)
    dpi_dtheta = jax.vmap(grad_pi_theta, in_axes=(0, None))(x_k, theta)

    # Scalar Hamiltonian derivative with respect to u:
    #     2 R pi(x_k,theta) + g^T lambda_{k+1}.
    h_u = 2.0 * R * u_k + lambda_k_plus_1 @ g

    return jnp.sum(dpi_dtheta * h_u[:, None], axis=0)


def trajectory_cost(states_full: jax.Array, theta: jax.Array) -> jax.Array:
    stage_costs = jax.vmap(stage_cost, in_axes=(0, None))(
        states_full[:-1],
        theta,
    )
    return jnp.sum(stage_costs) + terminal_cost(states_full[-1])


def two_pass_deer_gradient_single(
    x_0: jax.Array,
    theta: jax.Array,
    key: jax.Array,
):
    """State DEER -> costate DEER -> analytical policy gradient."""
    key_state, key_costate = jr.split(key)

    states_full, forward_steps = forward_deer_single(x_0, theta, key_state)
    costates_full, backward_steps = backward_deer_single(
        states_full,
        theta,
        key_costate,
    )

    gradient = policy_gradient(states_full, costates_full, theta)
    total_cost = trajectory_cost(states_full, theta)

    return gradient, total_cost, forward_steps, backward_steps


# vmap parallelizes independent Monte Carlo trajectories. DEER parallelizes the
# time direction inside each trajectory.
batched_two_pass_deer_gradient_single = jax.jit(
    jax.vmap(
        two_pass_deer_gradient_single,
        in_axes=(0, None, 0),
        out_axes=(0, 0, 0, 0),
    )
)


def monte_carlo_deer_gradient(
    theta: jax.Array,
    x_0_samples: jax.Array,
    key: jax.Array,
):
    sample_count = int(x_0_samples.shape[0])
    sample_keys = jr.split(key, sample_count)

    gradients, costs, forward_steps, backward_steps = (
        batched_two_pass_deer_gradient_single(
            x_0_samples,
            theta,
            sample_keys,
        )
    )

    mean_gradient = jnp.mean(gradients, axis=0)
    mean_cost = jnp.mean(costs)
    mean_forward_steps = jnp.mean(jnp.asarray(forward_steps, dtype=jnp.float64))
    mean_backward_steps = jnp.mean(jnp.asarray(backward_steps, dtype=jnp.float64))

    return (
        mean_gradient,
        mean_cost,
        mean_forward_steps,
        mean_backward_steps,
    )


# =============================================================================
# Euclidean projected-gradient update
# =============================================================================
def projected_gradient_update(
    theta: jax.Array,
    mean_gradient: jax.Array,
    condition_functions: Sequence[ConditionFunction],
    condition_gradient_functions: Sequence[ConditionGradientFunction],
    *,
    learning_rate: float = LEARNING_RATE,
    feasibility_margin: float = FEASIBILITY_MARGIN,
    projection_ftol: float = PROJECTION_FTOL,
    projection_max_iters: int = PROJECTION_MAX_ITERS,
) -> jax.Array:
    """
    Perform one Euclidean projected-gradient update.

    The constrained problem is assumed to have the form

        minimize    J(theta)
        subject to  h_i(theta) >= feasibility_margin.

    First take the unconstrained policy-gradient step

        theta_trial = theta - learning_rate * mean_gradient,

    and then solve the projection problem

        minimize_z  0.5 * ||z - theta_trial||_2^2
        subject to  h_i(z) >= feasibility_margin.

    For this controller, the feasible set is convex: the first, third,
    fourth, and fifth conditions are half-spaces, while

        a_1 * a_3 - a_2**2 >= feasibility_margin,  a_1 > 0

    is the epigraph of the convex quadratic-over-linear function

        a_3 >= (a_2**2 + feasibility_margin) / a_1.

    Consequently, the strictly convex projection objective has a unique
    solution. SLSQP is used only for this five-dimensional projection; the
    DEER trajectories and policy gradient remain in JAX.

    Parameters
    ----------
    theta:
        Current policy parameter.

    mean_gradient:
        Gradient of the original objective J with respect to theta.

    condition_functions:
        List of scalar functions h_i(theta). A parameter is feasible when
        every h_i(theta) is at least feasibility_margin.

    condition_gradient_functions:
        List containing grad h_i(theta), supplied to the projection solver.

    learning_rate:
        Policy-gradient step size.

    feasibility_margin:
        Closed-set margin imposed on every condition.

    projection_ftol:
        Termination tolerance for the Euclidean projection solve.

    projection_max_iters:
        Maximum number of projection iterations.

    Returns
    -------
    jax.Array
        The projected policy parameter.
    """
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if feasibility_margin <= 0.0:
        raise ValueError(
            "feasibility_margin must be positive because the controller "
            "contains divisions by constrained quantities."
        )

    if projection_ftol <= 0.0:
        raise ValueError("projection_ftol must be positive.")

    if projection_max_iters <= 0:
        raise ValueError("projection_max_iters must be positive.")

    feasibility_tolerance = max(10.0 * projection_ftol, 1.0e-10)

    if len(condition_functions) != len(condition_gradient_functions):
        raise ValueError(
            "condition_functions and condition_gradient_functions must have "
            "the same length."
        )

    theta_np = np.asarray(jax.device_get(theta), dtype=np.float64)
    gradient_np = np.asarray(
        jax.device_get(mean_gradient),
        dtype=np.float64,
    )

    if not np.all(np.isfinite(theta_np)):
        raise ValueError("theta contains a non-finite value.")
    if not np.all(np.isfinite(gradient_np)):
        raise ValueError("mean_gradient contains a non-finite value.")

    theta_trial = theta_np - learning_rate * gradient_np

    def condition_value(index: int, z: np.ndarray) -> float:
        value = np.asarray(
            jax.device_get(condition_functions[index](jnp.asarray(z)))
        )
        if value.ndim != 0:
            raise ValueError(
                f"Condition {index} must return a scalar, but returned "
                f"an array with shape {value.shape}."
            )
        return float(value) - feasibility_margin

    def condition_jacobian(index: int, z: np.ndarray) -> np.ndarray:
        return np.asarray(
            jax.device_get(
                condition_gradient_functions[index](jnp.asarray(z))
            ),
            dtype=np.float64,
        )

    # Avoid invoking the numerical solver when the ordinary gradient step is
    # already feasible. In that case the Euclidean projection is the point
    # itself.
    trial_constraint_values = np.asarray(
        [
            condition_value(i, theta_trial)
            for i in range(len(condition_functions))
        ]
    )
    if np.all(trial_constraint_values >= 0.0):
        return jnp.asarray(theta_trial, dtype=theta.dtype)

    current_constraint_values = np.asarray(
        [
            condition_value(i, theta_np)
            for i in range(len(condition_functions))
        ]
    )
    if np.any(current_constraint_values < -feasibility_tolerance):
        raise ValueError(
            "The current theta must be feasible before applying PGD. "
            f"Minimum shifted condition value: "
            f"{current_constraint_values.min():.6e}."
        )

    constraints = [
        {
            "type": "ineq",
            "fun": lambda z, i=i: condition_value(i, z),
            "jac": lambda z, i=i: condition_jacobian(i, z),
        }
        for i in range(len(condition_functions))
    ]

    def projection_objective(z: np.ndarray) -> float:
        displacement = z - theta_trial
        return 0.5 * float(displacement @ displacement)

    def projection_objective_gradient(z: np.ndarray) -> np.ndarray:
        return z - theta_trial

    projection_result = minimize(
        projection_objective,
        x0=theta_np,
        jac=projection_objective_gradient,
        constraints=constraints,
        method="SLSQP",
        options={
            "ftol": projection_ftol,
            "maxiter": projection_max_iters,
            "disp": False,
        },
    )

    projected_theta = np.asarray(projection_result.x, dtype=np.float64)
    projected_constraint_values = np.asarray(
        [
            condition_value(i, projected_theta)
            for i in range(len(condition_functions))
        ]
    )

    # SLSQP may report success with a residual on the order of its tolerance.
    # Reject a result that is materially infeasible instead of silently using
    # invalid controller parameters.
    if (
        not projection_result.success
        or not np.all(np.isfinite(projected_theta))
        or np.any(projected_constraint_values < -feasibility_tolerance)
    ):
        raise RuntimeError(
            "Euclidean projection failed. "
            f"Solver message: {projection_result.message}. "
            f"Minimum shifted condition value: "
            f"{projected_constraint_values.min():.6e}."
        )

    return jnp.asarray(projected_theta, dtype=theta.dtype)


# =============================================================================
# Final trajectory and visualization
# =============================================================================
def sequential_rollout(
    x_0: jax.Array,
    theta: jax.Array,
) -> jax.Array:
    """Sequential rollout used only for checking or plotting."""

    def scan_step(x_k, _):
        x_next = closed_loop_step(x_k, theta)
        return x_next, x_next

    _, states_tail = jax.lax.scan(
        scan_step,
        x_0,
        xs=None,
        length=T_HORIZON,
    )
    return jnp.vstack((x_0, states_tail))


def save_training_plots(
    cost_history: np.ndarray,
    parameter_history: np.ndarray,
    states: np.ndarray,
    costates: np.ndarray,
    controls: np.ndarray,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(cost_history)
    plt.xlabel("policy-gradient iteration")
    plt.ylabel("mean Monte Carlo cost")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mean_cost.png", dpi=180)
    plt.close()

    labels = [r"$a_1$", r"$a_2$", r"$a_3$", r"$k_p$", r"$k_v$"]
    plt.figure(figsize=(8, 5))
    for index, label in enumerate(labels):
        plt.plot(parameter_history[:, index], label=label)
    plt.xlabel("policy-gradient iteration")
    plt.ylabel("parameter value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parameter_history.png", dpi=180)
    plt.close()

    time_axis = DT * np.arange(T_HORIZON + 1)

    plt.figure(figsize=(8, 5))
    for index, label in enumerate([r"$q_1$", r"$q_2$", r"$p_1$", r"$p_2$"]):
        plt.plot(time_axis, states[:, index], label=label)
    plt.xlabel("time [s]")
    plt.ylabel("state")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_state_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for index, label in enumerate(
        [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
    ):
        plt.plot(time_axis, costates[:, index], label=label)
    plt.xlabel("time [s]")
    plt.ylabel("costate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_costate_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis[:-1], controls)
    plt.xlabel("time [s]")
    plt.ylabel(r"$u=\pi(x,\theta)$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "final_control.png", dpi=180)
    plt.close()


def save_gif(
    states: np.ndarray,
    controls: np.ndarray,
    output_file: Path,
) -> None:
    q_1 = states[:, 0]
    q_2 = states[:, 1]
    time_axis = DT * np.arange(T_HORIZON + 1)

    rod_length = 1.0
    wheel_radius = 0.18
    frame_step = max(1, int(0.05 / DT))
    frame_indices = np.arange(0, T_HORIZON + 1, frame_step)
    fps = 20

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.35)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Inertia-wheel pendulum: MC two-pass DEER + PGD")

    ax.plot(0.0, 0.0, "ko", markersize=8)
    rod, = ax.plot([], [], linewidth=4)
    wheel = Circle((0.0, 0.0), wheel_radius, fill=False, linewidth=3)
    ax.add_patch(wheel)
    spoke, = ax.plot([], [], linewidth=2)
    text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")

    def update(frame):
        index = frame_indices[frame]

        wheel_x = rod_length * np.sin(q_1[index])
        wheel_y = rod_length * np.cos(q_1[index])
        rod.set_data([0.0, wheel_x], [0.0, wheel_y])
        wheel.center = (wheel_x, wheel_y)

        dx = wheel_radius * np.sin(q_2[index])
        dy = wheel_radius * np.cos(q_2[index])
        spoke.set_data(
            [wheel_x - dx, wheel_x + dx],
            [wheel_y - dy, wheel_y + dy],
        )

        control_index = min(index, T_HORIZON - 1)
        text.set_text(
            f"t = {time_axis[index]:.2f} s\n"
            f"q_1 = {q_1[index]: .3f}\n"
            f"q_2 = {q_2[index]: .3f}\n"
            f"u = {controls[control_index]: .3f}"
        )
        return rod, wheel, spoke, text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / fps,
        blit=True,
    )
    animation.save(output_file, writer=PillowWriter(fps=fps))
    plt.close(fig)


# =============================================================================
# Main optimization loop
# =============================================================================
def main() -> None:
    master_key = jr.PRNGKey(SEED)
    key_samples, key_training, key_final = jr.split(master_key, 3)

    # Fixed Monte Carlo batch, matching the structure of the supplied example.
    x_0_samples = sample_initial_states(key_samples, NUM_MC_SAMPLES)
    theta = theta_initial

    cost_history = []
    gradient_norm_history = []
    parameter_history = []

    print("Inertia-wheel pendulum: two-pass DEER + Monte Carlo PGD")
    print(f"T_HORIZON={T_HORIZON}, NUM_MC_SAMPLES={NUM_MC_SAMPLES}")
    print(f"DEER_MAX_ITERS={DEER_MAX_ITERS}, LEARNING_RATE={LEARNING_RATE}")
    print("Initial theta [a_1, a_2, a_3, k_p, k_v]:")
    print(np.asarray(theta))
    print()

    start = time.perf_counter()

    for iteration in range(NUM_POLICY_ITERS):
        iteration_key = jr.fold_in(key_training, iteration)

        (
            mean_gradient,
            mean_cost,
            mean_forward_steps,
            mean_backward_steps,
        ) = monte_carlo_deer_gradient(
            theta,
            x_0_samples,
            iteration_key,
        )

        gradient_norm = jnp.linalg.norm(mean_gradient)

        # Project the unconstrained policy-gradient step onto the complete
        # feasible parameter set. No barrier, Adam, momentum, or adaptive
        # scaling is used.
        theta = projected_gradient_update(
            theta,
            mean_gradient,
            condition_functions,
            condition_gradient_functions,
        )

        cost_history.append(float(mean_cost))
        gradient_norm_history.append(float(gradient_norm))
        parameter_history.append(np.asarray(theta))

        if iteration % PRINT_EVERY == 0 or iteration == NUM_POLICY_ITERS - 1:
            print(
                f"iter={iteration:03d} | "
                f"mean_cost={float(mean_cost):12.6f} | "
                f"grad_norm={float(gradient_norm):10.4e} | "
                f"fwd_steps={float(mean_forward_steps):6.1f} | "
                f"bwd_steps={float(mean_backward_steps):6.1f}"
            )

    elapsed = time.perf_counter() - start
    print(f"\nTotal training time: {elapsed:.3f} seconds")
    print("Optimized theta [a_1, a_2, a_3, k_p, k_v]:")
    print(np.asarray(theta))

    # Evaluate one representative initial condition with the same two DEER passes.
    x_0_test = jnp.array([jnp.pi - 0.20, 0.0, 0.0, 0.0])
    key_state, key_costate = jr.split(key_final)
    final_states, _ = forward_deer_single(x_0_test, theta, key_state)
    final_costates, _ = backward_deer_single(
        final_states,
        theta,
        key_costate,
    )
    final_controls = jax.vmap(pi, in_axes=(0, None))(final_states[:-1], theta)

    (
        final_states_np,
        final_costates_np,
        final_controls_np,
        theta_np,
    ) = map(
        np.asarray,
        jax.device_get(
            (final_states, final_costates, final_controls, theta)
        ),
    )

    cost_history_np = np.asarray(cost_history)
    gradient_norm_history_np = np.asarray(gradient_norm_history)
    parameter_history_np = np.asarray(parameter_history)

    save_training_plots(
        cost_history_np,
        parameter_history_np,
        final_states_np,
        final_costates_np,
        final_controls_np,
    )

    gif_file = RESULTS_DIR / "inertia_wheel_two_pass_deer_mc.gif"
    save_gif(final_states_np, final_controls_np, gif_file)

    np.savetxt(
        RESULTS_DIR / "optimized_parameters.txt",
        theta_np[None, :],
        header="a_1 a_2 a_3 k_p k_v",
    )
    np.savez(
        RESULTS_DIR / "training_results.npz",
        theta=theta_np,
        x_0_samples=np.asarray(x_0_samples),
        cost_history=cost_history_np,
        gradient_norm_history=gradient_norm_history_np,
        parameter_history=parameter_history_np,
        final_states=final_states_np,
        final_costates=final_costates_np,
        final_controls=final_controls_np,
        time=DT * np.arange(T_HORIZON + 1),
    )

    print(f"Final test state: {final_states_np[-1]}")
    print(f"Saved results in: {RESULTS_DIR.resolve()}")
    print(f"Saved GIF: {gif_file.resolve()}")


if __name__ == "__main__":
    main()
