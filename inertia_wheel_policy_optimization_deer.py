"""Nonlinear policy optimization for the inertia-wheel pendulum in JAX.

The controller has the same IDA-PBC structure as the previous simulation:

    theta = [a_1, a_2, a_3, k_p, k_v]

    gamma_1 = (a_2 / (a_1 + a_2)) m_3
    gamma_2 = -m_11 (a_2 + a_3) / (m_22 (a_1 + a_2))
    k_2     = -m_22 (a_1 + a_2) / (a_1 a_3 - a_2^2)

    pi(x, theta)
      = gamma_1 sin(q_1)
        + k_p (q_2 + gamma_2 q_1)
        + k_v k_2 (q_2_dot + gamma_2 q_1_dot)

The continuous-time plant is discretized with forward Euler so that

    x_{k+1} = f(x_k) + g pi(x_k, theta),

and the requested policy-gradient formula is exact for the discrete model:

    grad_theta J
      = sum_k (grad_theta pi_k)^T
          [2 R pi_k + g^T lambda_{k+1}].

Both state and costate recurrences are evaluated with a compact DEER
implementation. DEER applies Newton linearization to the full nonlinear
recurrence and solves each linearized affine recurrence with
jax.lax.associative_scan.

Running this file:
  1. optimizes a_1, a_2, a_3, k_p, k_v;
  2. verifies the hand-written policy gradient against JAX autodiff;
  3. saves optimization and trajectory plots;
  4. saves inertia_wheel_policy_optimization_deer.gif.

Dependencies:
    pip install "jax[cpu]" matplotlib pillow
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

jax.config.update("jax_enable_x64", True)

Array = jax.Array


# =============================================================================
# Inertia-wheel pendulum parameters
# =============================================================================
m_11 = 0.1
m_22 = 0.2
m_3 = 10.0

# Discrete-time horizon used to approximate the infinite sum.
dt = 0.04
t_f = 8.0
N = int(t_f / dt)

# x = [q_1, q_2, p_1, p_2].
x_0 = jnp.array([3.14, 0.0, 0.0, 0.0])

# Continuous input vector and its discrete-time counterpart.
g_c = jnp.array([0.0, 0.0, -1.0, 1.0])
g = dt * g_c

# Discrete running and terminal costs.
# Multiplication by dt makes the sum approximate a continuous-time integral.
Q = dt * jnp.diag(jnp.array([12.0, 0.15, 0.20, 0.08]))
R = dt * 2.0e-3
Q_f = jnp.diag(jnp.array([40.0, 0.50, 1.00, 0.30]))

# Initial values from the paper example.
theta_0 = jnp.array([1.0, -1.5, 6.0, 3.75, 10.0])


# =============================================================================
# Controller and closed-loop system
# =============================================================================
def controller_constants(theta: Array) -> tuple[Array, Array, Array]:
    """Return gamma_1, gamma_2, and k_2 from theta."""
    a_1, a_2, a_3, _, _ = theta

    gamma_1 = (a_2 / (a_1 + a_2)) * m_3
    gamma_2 = -m_11 * (a_2 + a_3) / (m_22 * (a_1 + a_2))
    k_2 = -m_22 * (a_1 + a_2) / (a_1 * a_3 - a_2**2)
    return gamma_1, gamma_2, k_2


def pi(x: Array, theta: Array) -> Array:
    """IDA-PBC policy pi(x, theta)."""
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


def f(x: Array) -> Array:
    """Unforced discrete dynamics f(x) = x + dt f_c(x)."""
    q_1, q_2, p_1, p_2 = x
    f_c = jnp.array(
        [
            p_1 / m_11,
            p_2 / m_22,
            m_3 * jnp.sin(q_1),
            0.0,
        ]
    )
    return x + dt * f_c


def closed_loop_step(x: Array, theta: Array) -> Array:
    """Closed-loop recurrence x_{k+1} = f(x_k) + g pi(x_k, theta)."""
    return f(x) + g * pi(x, theta)


def stage_cost(x: Array, theta: Array) -> Array:
    u = pi(x, theta)
    return x @ Q @ x + R * u**2


def terminal_cost(x: Array) -> Array:
    return x @ Q_f @ x


# =============================================================================
# Parallel affine scan
# =============================================================================
def _compose_affine(
    left: tuple[Array, Array], right: tuple[Array, Array]
) -> tuple[Array, Array]:
    """Compose right(left(x)) for affine maps represented by (A, b)."""
    A_l, b_l = left
    A_r, b_r = right
    A = jnp.einsum("...ij,...jk->...ik", A_r, A_l)
    b = jnp.einsum("...ij,...j->...i", A_r, b_l) + b_r
    return A, b


def parallel_affine_rollout(A: Array, b: Array, initial: Array) -> Array:
    """Solve y_{k+1} = A_k y_k + b_k with an associative scan."""
    A_prefix, b_prefix = jax.lax.associative_scan(_compose_affine, (A, b))
    return jnp.einsum("kij,j->ki", A_prefix, initial) + b_prefix


# =============================================================================
# DEER nonlinear recurrence solver
# =============================================================================
def deer_solve(
    recurrence,
    initial: Array,
    guess: Array,
    indices: Array,
    num_iterations: int,
) -> Array:
    """Evaluate y_{k+1}=recurrence(y_k,k) using DEER/Newton iterations.

    Args:
        recurrence: function (y_k, k) -> y_{k+1}.
        initial: fixed y_0.
        guess: initial guess for [y_1, ..., y_N].
        indices: recurrence indices [0, ..., N-1].
        num_iterations: fixed number of Newton/DEER refinements.

    Returns:
        Array [y_0, y_1, ..., y_N].
    """
    jac_recurrence = jax.jacrev(recurrence, argnums=0)

    def newton_iteration(_, current_guess):
        previous_guess = jnp.concatenate((initial[None, :], current_guess[:-1]), axis=0)

        A = jax.vmap(jac_recurrence)(previous_guess, indices)
        recurrence_value = jax.vmap(recurrence)(previous_guess, indices)
        b = recurrence_value - jnp.einsum("kij,kj->ki", A, previous_guess)

        return parallel_affine_rollout(A, b, initial)

    solution = jax.lax.fori_loop(0, num_iterations, newton_iteration, guess)
    return jnp.concatenate((initial[None, :], solution), axis=0)


# A modest number is sufficient when warm-starting from the preceding optimizer
# iteration. Increase these if DEER residuals are not small on your hardware.
STATE_DEER_ITERATIONS = 6
COSTATE_DEER_ITERATIONS = 2  # the costate recurrence is affine, so one is enough
indices = jnp.arange(N)


def sequential_state_rollout(theta: Array) -> Array:
    """Only used once to construct a reliable initial DEER guess."""

    def scan_step(x, _):
        x_next = closed_loop_step(x, theta)
        return x_next, x_next

    _, tail = jax.lax.scan(scan_step, x_0, xs=None, length=N)
    return jnp.concatenate((x_0[None, :], tail), axis=0)


def state_trajectory_deer(theta: Array, guess: Array) -> Array:
    recurrence = lambda x, _: closed_loop_step(x, theta)
    return deer_solve(
        recurrence=recurrence,
        initial=x_0,
        guess=guess,
        indices=indices,
        num_iterations=STATE_DEER_ITERATIONS,
    )


grad_stage_x = jax.grad(stage_cost, argnums=0)
jac_closed_loop_x = jax.jacrev(closed_loop_step, argnums=0)
grad_terminal_x = jax.grad(terminal_cost)


def costate_trajectory_deer(
    states: Array, theta: Array, reverse_guess: Array
) -> Array:
    """Return lambda_0,...,lambda_N using a reversed-time DEER solve."""
    lambda_N = grad_terminal_x(states[-1])

    # mu_j = lambda_{N-j}. The recurrence advances forward in j.
    def reverse_costate_step(mu: Array, j: Array) -> Array:
        k = N - 1 - j
        x_k = states[k]
        return grad_stage_x(x_k, theta) + jac_closed_loop_x(x_k, theta).T @ mu

    mu = deer_solve(
        recurrence=reverse_costate_step,
        initial=lambda_N,
        guess=reverse_guess,
        indices=indices,
        num_iterations=COSTATE_DEER_ITERATIONS,
    )

    return jnp.flip(mu, axis=0)


# =============================================================================
# Requested nonlinear policy gradient
# =============================================================================
grad_pi_theta = jax.jacrev(pi, argnums=1)


def policy_gradient(states: Array, costates: Array, theta: Array) -> Array:
    """Compute the requested analytical nonlinear policy gradient."""
    x_k = states[:-1]
    lambda_next = costates[1:]

    u_k = jax.vmap(pi, in_axes=(0, None))(x_k, theta)
    dpi_dtheta = jax.vmap(grad_pi_theta, in_axes=(0, None))(x_k, theta)

    h_u = 2.0 * R * u_k + lambda_next @ g
    return jnp.sum(dpi_dtheta * h_u[:, None], axis=0)


def trajectory_cost(states: Array, theta: Array) -> Array:
    running = jnp.sum(jax.vmap(stage_cost, in_axes=(0, None))(states[:-1], theta))
    return running + terminal_cost(states[-1])


def deer_residual(states: Array, theta: Array) -> Array:
    predicted = jax.vmap(closed_loop_step, in_axes=(0, None))(states[:-1], theta)
    return jnp.max(jnp.linalg.norm(states[1:] - predicted, axis=1))


# =============================================================================
# Projected Adam optimizer
# =============================================================================
def project_theta(theta: Array) -> Array:
    """Keep the IDA-PBC formulas away from singular parameter choices."""
    a_1, a_2, a_3, k_p, k_v = theta

    a_1 = jnp.clip(a_1, 0.10, 8.0)
    a_2 = jnp.clip(a_2, -6.0, -0.10)

    # Preserve a_1 + a_2 < 0 and keep it separated from zero.
    a_2 = jnp.minimum(a_2, -a_1 - 0.10)

    # Positive-definite desired inertia matrix:
    # a_1 > 0 and a_1 a_3 - a_2^2 > 0.
    a_3_min = a_2**2 / a_1 + 0.10
    a_3 = jnp.clip(jnp.maximum(a_3, a_3_min), 0.20, 40.0)

    k_p = jnp.clip(k_p, 0.02, 20.0)
    k_v = jnp.clip(k_v, 0.02, 30.0)

    return jnp.array([a_1, a_2, a_3, k_p, k_v])


def adam_update(
    theta: Array,
    gradient: Array,
    first_moment: Array,
    second_moment: Array,
    iteration: Array,
    learning_rate: float,
) -> tuple[Array, Array, Array]:
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1.0e-8

    first_moment = beta_1 * first_moment + (1.0 - beta_1) * gradient
    second_moment = beta_2 * second_moment + (1.0 - beta_2) * gradient**2

    m_hat = first_moment / (1.0 - beta_1 ** (iteration + 1))
    v_hat = second_moment / (1.0 - beta_2 ** (iteration + 1))

    theta = theta - learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
    return project_theta(theta), first_moment, second_moment


@partial(jax.jit, static_argnames=("learning_rate",))
def optimization_step(
    theta: Array,
    state_guess: Array,
    reverse_costate_guess: Array,
    first_moment: Array,
    second_moment: Array,
    iteration: Array,
    learning_rate: float,
):
    states = state_trajectory_deer(theta, state_guess)
    costates = costate_trajectory_deer(states, theta, reverse_costate_guess)

    gradient = policy_gradient(states, costates, theta)

    # The formula is a sum. Dividing by N only rescales the optimizer step and
    # leaves the minimizer unchanged. Gradient clipping improves robustness.
    optimizer_gradient = gradient / N
    gradient_norm = jnp.linalg.norm(optimizer_gradient)
    optimizer_gradient *= jnp.minimum(1.0, 5.0 / (gradient_norm + 1.0e-12))

    theta_next, first_moment, second_moment = adam_update(
        theta,
        optimizer_gradient,
        first_moment,
        second_moment,
        iteration,
        learning_rate,
    )

    reverse_costates = jnp.flip(costates, axis=0)[1:]

    return (
        theta_next,
        states[1:],
        reverse_costates,
        first_moment,
        second_moment,
        trajectory_cost(states, theta),
        jnp.linalg.norm(gradient),
        deer_residual(states, theta),
    )


# =============================================================================
# Gradient verification
# =============================================================================
def sequential_cost(theta: Array) -> Array:
    return trajectory_cost(sequential_state_rollout(theta), theta)


def verify_gradient(theta: Array, states: Array, costates: Array) -> tuple[Array, Array, Array]:
    manual = policy_gradient(states, costates, theta)
    autodiff = jax.grad(sequential_cost)(theta)
    relative_error = jnp.linalg.norm(manual - autodiff) / (
        jnp.linalg.norm(autodiff) + 1.0e-12
    )
    return manual, autodiff, relative_error


# =============================================================================
# Plotting and GIF
# =============================================================================
def save_plots(
    output_directory: Path,
    costs: np.ndarray,
    parameter_history: np.ndarray,
    states: np.ndarray,
    costates: np.ndarray,
    controls: np.ndarray,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.semilogy(costs)
    plt.xlabel("optimization iteration")
    plt.ylabel("J(x_0, theta)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_directory / "optimization_cost.png", dpi=180)
    plt.close()

    labels = [r"$a_1$", r"$a_2$", r"$a_3$", r"$k_p$", r"$k_v$"]
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(parameter_history[:, i], label=label)
    plt.xlabel("optimization iteration")
    plt.ylabel("parameter value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "parameter_history.png", dpi=180)
    plt.close()

    time = np.linspace(0.0, t_f, N + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(time, states[:, 0], label=r"$q_1$")
    plt.plot(time, states[:, 1], label=r"$q_2$")
    plt.plot(time, states[:, 2], label=r"$p_1$")
    plt.plot(time, states[:, 3], label=r"$p_2$")
    plt.xlabel("time [s]")
    plt.ylabel("state")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "optimized_state_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for i, label in enumerate([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]):
        plt.plot(time, costates[:, i], label=label)
    plt.xlabel("time [s]")
    plt.ylabel("costate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory / "optimized_costate_trajectory.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(time[:-1], controls)
    plt.xlabel("time [s]")
    plt.ylabel(r"$u=\pi(x,\theta)$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_directory / "optimized_control.png", dpi=180)
    plt.close()


def save_gif(output_file: Path, states: np.ndarray, controls: np.ndarray) -> None:
    time = np.linspace(0.0, t_f, N + 1)
    q_1 = states[:, 0]
    q_2 = states[:, 1]

    rod_length = 1.0
    wheel_radius = 0.18
    fps = 20
    frame_step = max(1, int(0.16 / dt))
    frame_indices = np.arange(0, N + 1, frame_step)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.35)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimized Inertia-Wheel Pendulum: DEER")

    ax.plot(0.0, 0.0, "ko", markersize=8)
    (rod,) = ax.plot([], [], linewidth=4)
    wheel = Circle((0.0, 0.0), wheel_radius, fill=False, linewidth=3)
    ax.add_patch(wheel)
    (spoke,) = ax.plot([], [], linewidth=2)
    text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        i = frame_indices[frame]
        wheel_x = rod_length * np.sin(q_1[i])
        wheel_y = rod_length * np.cos(q_1[i])

        rod.set_data([0.0, wheel_x], [0.0, wheel_y])
        wheel.center = (wheel_x, wheel_y)

        dx = wheel_radius * np.sin(q_2[i])
        dy = wheel_radius * np.cos(q_2[i])
        spoke.set_data(
            [wheel_x - dx, wheel_x + dx],
            [wheel_y - dy, wheel_y + dy],
        )

        control_index = min(i, N - 1)
        text.set_text(
            f"t = {time[i]:5.2f} s\n"
            f"q_1 = {q_1[i]: .3f} rad\n"
            f"q_2 = {q_2[i]: .3f} rad\n"
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
# Main
# =============================================================================
def main() -> None:
    output_directory = Path("inertia_wheel_deer_results")
    output_directory.mkdir(parents=True, exist_ok=True)

    num_optimizer_steps = 35
    learning_rate = 1.0e-2
    print_every = 5

    theta = project_theta(theta_0)

    # One sequential rollout supplies the first Newton guess. Every later DEER
    # solve is warm-started by the preceding optimizer iteration's trajectory.
    initial_states = sequential_state_rollout(theta)
    state_guess = initial_states[1:]

    initial_costates = costate_trajectory_deer(
        initial_states,
        theta,
        jnp.zeros((N, 4), dtype=theta.dtype),
    )
    reverse_costate_guess = jnp.flip(initial_costates, axis=0)[1:]

    first_moment = jnp.zeros_like(theta)
    second_moment = jnp.zeros_like(theta)

    cost_history = []
    parameter_history = []
    best_cost = np.inf
    best_theta_np = np.asarray(jax.device_get(theta)).copy()

    print("Initial theta [a_1, a_2, a_3, k_p, k_v]:")
    print(best_theta_np)

    for iteration in range(num_optimizer_steps):
        theta_used_np = np.asarray(jax.device_get(theta)).copy()
        (
            theta,
            state_guess,
            reverse_costate_guess,
            first_moment,
            second_moment,
            cost,
            gradient_norm,
            residual,
        ) = optimization_step(
            theta,
            state_guess,
            reverse_costate_guess,
            first_moment,
            second_moment,
            jnp.asarray(iteration),
            learning_rate,
        )

        cost_value, gradient_norm_value, residual_value = jax.device_get(
            (cost, gradient_norm, residual)
        )
        cost_history.append(float(cost_value))
        parameter_history.append(theta_used_np)

        if cost_value < best_cost:
            best_cost = float(cost_value)
            best_theta_np = theta_used_np.copy()

        if iteration % print_every == 0 or iteration == num_optimizer_steps - 1:
            print(
                f"iteration {iteration:4d} | "
                f"J = {cost_value:12.6f} | "
                f"|grad J| = {gradient_norm_value:10.3e} | "
                f"DEER residual = {residual_value:10.3e}"
            )

    theta = jnp.asarray(best_theta_np)

    # Final DEER state and costate trajectories for the best policy.
    final_states = jax.jit(state_trajectory_deer)(theta, state_guess)
    final_costates = jax.jit(costate_trajectory_deer)(
        final_states, theta, reverse_costate_guess
    )
    final_controls = jax.jit(
        lambda states, parameters: jax.vmap(pi, in_axes=(0, None))(
            states[:-1], parameters
        )
    )(final_states, theta)

    manual_gradient, autodiff_gradient, relative_error = verify_gradient(
        theta, final_states, final_costates
    )

    theta_np, states_np, costates_np, controls_np = jax.device_get(
        (theta, final_states, final_costates, final_controls)
    )
    gamma_1, gamma_2, k_2 = jax.device_get(controller_constants(theta))

    cost_history_np = np.asarray(cost_history)
    parameter_history_np = np.asarray(parameter_history)

    print(f"\nBest objective value: {best_cost:.6f}")
    print("Optimized theta [a_1, a_2, a_3, k_p, k_v]:")
    print(theta_np)
    print("Controller constants [gamma_1, gamma_2, k_2]:")
    print(np.array([gamma_1, gamma_2, k_2]))
    print("Final state [q_1, q_2, p_1, p_2]:")
    print(states_np[-1])
    print("Manual policy gradient:")
    print(np.asarray(jax.device_get(manual_gradient)))
    print("JAX autodiff gradient:")
    print(np.asarray(jax.device_get(autodiff_gradient)))
    print(f"Relative gradient error: {float(relative_error):.3e}")

    save_plots(
        output_directory,
        cost_history_np,
        parameter_history_np,
        states_np,
        costates_np,
        controls_np,
    )

    np.savetxt(
        output_directory / "optimized_parameters.txt",
        theta_np[None, :],
        header="a_1 a_2 a_3 k_p k_v",
    )
    np.savez(
        output_directory / "optimized_trajectory.npz",
        theta=theta_np,
        states=states_np,
        costates=costates_np,
        controls=controls_np,
        time=np.linspace(0.0, t_f, N + 1),
        cost_history=cost_history_np,
        parameter_history=parameter_history_np,
    )

    gif_file = output_directory / "inertia_wheel_policy_optimization_deer.gif"
    save_gif(gif_file, states_np, controls_np)

    print(f"\nSaved results in: {output_directory.resolve()}")
    print(f"Saved GIF: {gif_file.resolve()}")


if __name__ == "__main__":
    main()
