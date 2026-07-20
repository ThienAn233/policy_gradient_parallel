"""
iwp_ida_pbc_deer_policy_iteration.py

Inertia-Wheel Pendulum simulation + IDA-PBC stabilizing controller
+ DEER/costate policy-gradient parameter optimization.

System:
    x = [q1, q2, p1, p2]

    q1: pendulum angle, q1=0 is upright
    q2: wheel angle
    p1, p2: momenta-like coordinates

Hamiltonian-coordinate model used in IDA-PBC literature:

    q1_dot = p1 / m11
    q2_dot = p2 / m22
    p1_dot = m3 sin(q1) - u
    p2_dot = u

IDA-PBC bounded stabilizing controller form:

    u = gamma1 sin(q1)
        + kp tanh(q2 + gamma2 q1)
        + kv tanh(k3 (p2 + k4 p1))

where:
    gamma1, gamma2, k3, k4 come from matching/design,
    kp, kv > 0 are tunable gains.

This script optimizes kp and kv with DEER-based policy gradient.
The parameterization enforces the bounded-controller condition

    gamma1 + kp + kv <= u_max.

Requires:
    deer.py with deer_alg
"""

from pathlib import Path
from typing import NamedTuple

import time

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from deer import deer_alg


# ============================================================
# 1. Parameters
# ============================================================

class IWPPhysicalParams(NamedTuple):
    # Hamiltonian inertia parameters
    m11: float = 1.0
    m22: float = 0.20
    m3: float = 1.0

    # Integrator/control limits
    dt: float = 0.02
    u_max: float = 5.0


class IDAStructureParams(NamedTuple):
    # IDA-PBC matching/controller constants.
    # Choose gamma1 > m3 for local upright stabilization.
    gamma1: float = 1.50
    gamma2: float = 1.00
    k3: float = 1.00
    k4: float = 1.00


class CostWeights(NamedTuple):
    q1: float = 10.0
    z: float = 2.0
    p1: float = 0.10
    p2: float = 0.05
    u: float = 0.001

    terminal_q1: float = 50.0
    terminal_z: float = 10.0
    terminal_p1: float = 1.0
    terminal_p2: float = 0.5


# ============================================================
# 2. Global settings
# ============================================================

SEED = 0

T_HORIZON = 300
BATCH_SIZE = 16
NUM_POLICY_ITERS = 80

DEER_MAX_ITERS = 5
DEER_TOL = 1e-7

LEARNING_RATE = 5e-2

RESULT_DIR = Path("iwp_ida_pbc_deer_policy_iteration_results")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

STATE_DIM = 4
ACTION_DIM = 1


# ============================================================
# 3. Math helpers
# ============================================================

def wrap_to_pi(angle):
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def raw_to_kp_kv(raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams):
    """
    Convert raw parameters to positive bounded-controller gains.

    The bounded controller has

        |u| <= gamma1 + kp + kv.

    This function enforces

        gamma1 + kp + kv < u_max

    by splitting the remaining torque budget with a softmax.
    """
    torque_budget = phys.u_max - ida.gamma1
    torque_budget = jnp.maximum(torque_budget - 1e-4, 1e-4)

    # Three-way softmax:
    #   fraction 0 -> kp
    #   fraction 1 -> kv
    #   fraction 2 -> unused slack
    fractions = jax.nn.softmax(jnp.array([raw_gains[0], raw_gains[1], 0.0]))

    kp = torque_budget * fractions[0]
    kv = torque_budget * fractions[1]

    return kp, kv


def make_raw_gains_from_kp_kv(kp, kv, phys: IWPPhysicalParams, ida: IDAStructureParams):
    """
    Convenience initializer.

    Approximately initializes the softmax parameterization so that the
    resulting kp, kv are close to the requested values.
    """
    budget = phys.u_max - ida.gamma1 - 1e-4
    budget = max(float(budget), 1e-4)

    kp = min(float(kp), 0.45 * budget)
    kv = min(float(kv), 0.45 * budget)

    slack = max(budget - kp - kv, 1e-4)

    raw_kp = np.log(kp / slack)
    raw_kv = np.log(kv / slack)

    return jnp.array([raw_kp, raw_kv])


# ============================================================
# 4. IDA-PBC controller
# ============================================================

def ida_pbc_controller(x, raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams):
    """
    Bounded IDA-PBC-style stabilizing controller:

        u = gamma1 sin(q1)
            + kp tanh(q2 + gamma2 q1)
            + kv tanh(k3 (p2 + k4 p1))

    Energy shaping:
        gamma1 sin(q1) + kp tanh(q2 + gamma2 q1)

    Damping injection:
        kv tanh(k3 (p2 + k4 p1))
    """
    q1, q2, p1, p2 = x

    kp, kv = raw_to_kp_kv(raw_gains, phys, ida)

    z = q2 + ida.gamma2 * q1
    damping_coord = ida.k3 * (p2 + ida.k4 * p1)

    u_es = ida.gamma1 * jnp.sin(q1) + kp * jnp.tanh(z)
    u_di = kv * jnp.tanh(damping_coord)

    u = u_es + u_di

    return jnp.clip(u, -phys.u_max, phys.u_max)


# ============================================================
# 5. Dynamics and costs
# ============================================================

def continuous_dynamics(x, raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams):
    """
    Continuous-time inertia-wheel pendulum model.

    x = [q1, q2, p1, p2]
    """
    q1, q2, p1, p2 = x

    u = ida_pbc_controller(x, raw_gains, phys, ida)

    q1_dot = p1 / phys.m11
    q2_dot = p2 / phys.m22
    p1_dot = phys.m3 * jnp.sin(q1) - u
    p2_dot = u

    return jnp.array([q1_dot, q2_dot, p1_dot, p2_dot])


def rk4_step(x, raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams):
    dt = phys.dt

    k1 = continuous_dynamics(x, raw_gains, phys, ida)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, raw_gains, phys, ida)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, raw_gains, phys, ida)
    k4 = continuous_dynamics(x + dt * k3, raw_gains, phys, ida)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # q1 is an angle around upright. q2 is the wheel coordinate, so keep it unwrapped.
    return jnp.array([wrap_to_pi(x_next[0]), x_next[1], x_next[2], x_next[3]])


def stage_cost(x, raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams, weights: CostWeights):
    q1, q2, p1, p2 = x

    q1_error = wrap_to_pi(q1)
    z = q2 + ida.gamma2 * q1
    u = ida_pbc_controller(x, raw_gains, phys, ida)

    return (
        weights.q1 * q1_error**2
        + weights.z * z**2
        + weights.p1 * p1**2
        + weights.p2 * p2**2
        + weights.u * u**2
    )


def terminal_cost(x_T, raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams, weights: CostWeights):
    """State-only terminal cost."""
    q1, q2, p1, p2 = x_T

    q1_error = wrap_to_pi(q1)
    z = q2 + ida.gamma2 * q1

    return (
        weights.terminal_q1 * q1_error**2
        + weights.terminal_z * z**2
        + weights.terminal_p1 * p1**2
        + weights.terminal_p2 * p2**2
    )


# ============================================================
# 6. Sequential rollout for guesses and visualization
# ============================================================

def rollout_closed_loop(x0, raw_gains, phys: IWPPhysicalParams, ida: IDAStructureParams, weights: CostWeights):
    def step(x, _):
        u = ida_pbc_controller(x, raw_gains, phys, ida)
        c = stage_cost(x, raw_gains, phys, ida, weights)
        x_next = rk4_step(x, raw_gains, phys, ida)
        return x_next, (x, u, c)

    x_T, (xs, us, costs) = jax.lax.scan(step, x0, xs=None, length=T_HORIZON)
    xs_full = jnp.vstack([xs, x_T[None, :]])
    return xs_full, us, costs


# ============================================================
# 7. DEER two-pass costate gradient for one sample
# ============================================================

def deer_policy_gradient_single(
    x0,
    raw_gains,
    key,
    phys: IWPPhysicalParams,
    ida: IDAStructureParams,
    weights: CostWeights,
):
    """
    Compute gradient wrt raw_gains using:

        1. Forward DEER for x_1,...,x_T
        2. Backward DEER for lambda_{T-1},...,lambda_0
        3. Hamiltonian parameter gradient sum
    """
    key_x, key_lam = jr.split(key)
    dummy_inputs = jnp.zeros((T_HORIZON, ACTION_DIM))

    # Forward DEER
    def forward_f(x, dummy):
        return rk4_step(x, raw_gains, phys, ida)

    # Good initial guess: sequential rollout. This makes nonlinear DEER stable.
    x_guess_full, _, _ = rollout_closed_loop(x0, raw_gains, phys, ida, weights)
    x_guess = x_guess_full[1:]
    x_guess = x_guess + 1e-6 * jr.normal(key_x, shape=x_guess.shape)

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

    x_states = forward_result[1]  # [x_1, ..., x_T]
    x_T = x_states[-1]
    x_traj = jnp.vstack([x0, x_states[:-1]])  # [x_0, ..., x_{T-1}]

    lambda_T = jax.grad(lambda x: terminal_cost(x, raw_gains, phys, ida, weights))(x_T)

    # Backward DEER
    def backward_f(lambda_next, x_k):
        l_x = jax.grad(lambda x: stage_cost(x, raw_gains, phys, ida, weights))(x_k)
        F_x = jax.jacrev(lambda x: rk4_step(x, raw_gains, phys, ida))(x_k)
        return l_x + F_x.T @ lambda_next

    lambda_guess = 1e-6 * jr.normal(key_lam, shape=(T_HORIZON, STATE_DIM))
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
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)
    lambda_next = jnp.vstack([lambda_chronological[1:], lambda_T[None, :]])

    # Parameter gradient from Hamiltonian terms
    def hamiltonian_grad(x_k, lambda_k_plus_1):
        def H(raw):
            return (
                stage_cost(x_k, raw, phys, ida, weights)
                + jnp.dot(lambda_k_plus_1, rk4_step(x_k, raw, phys, ida))
            )
        return jax.grad(H)(raw_gains)

    grad_terms = jax.vmap(hamiltonian_grad)(x_traj, lambda_next)
    grad_raw = jnp.sum(grad_terms, axis=0)

    total_cost = (
        jnp.sum(jax.vmap(lambda x: stage_cost(x, raw_gains, phys, ida, weights))(x_traj))
        + terminal_cost(x_T, raw_gains, phys, ida, weights)
    )

    return grad_raw, total_cost


# Batch vmap across initial conditions
batched_deer_policy_gradient = jax.jit(
    jax.vmap(
        deer_policy_gradient_single,
        in_axes=(0, None, 0, None, None, None),
        out_axes=(0, 0),
    )
)


def monte_carlo_deer_gradient(
    raw_gains,
    x0_batch,
    key,
    phys: IWPPhysicalParams,
    ida: IDAStructureParams,
    weights: CostWeights,
):
    keys = jr.split(key, x0_batch.shape[0])
    grad_batch, cost_batch = batched_deer_policy_gradient(x0_batch, raw_gains, keys, phys, ida, weights)
    mean_grad = jnp.mean(grad_batch, axis=0)
    mean_cost = jnp.mean(cost_batch)
    return mean_grad, mean_cost


# ============================================================
# 8. Adam optimizer
# ============================================================

class AdamState(NamedTuple):
    m: jnp.ndarray
    v: jnp.ndarray
    t: int


def adam_init(params):
    return AdamState(m=jnp.zeros_like(params), v=jnp.zeros_like(params), t=0)


def adam_update(params, grad, state: AdamState, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
    t = state.t + 1
    m = beta1 * state.m + (1.0 - beta1) * grad
    v = beta2 * state.v + (1.0 - beta2) * (grad**2)
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    params_next = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params_next, AdamState(m=m, v=v, t=t)


# ============================================================
# 9. Sampling
# ============================================================

def sample_initial_conditions(key, batch_size):
    """Initial states near the hanging position q1=pi."""
    k1, k2, k3, k4 = jr.split(key, 4)
    q1 = jnp.pi + jr.uniform(k1, (batch_size,), minval=-0.4, maxval=0.4)
    q2 = jr.uniform(k2, (batch_size,), minval=-0.2, maxval=0.2)
    p1 = jr.uniform(k3, (batch_size,), minval=-0.2, maxval=0.2)
    p2 = jr.uniform(k4, (batch_size,), minval=-0.2, maxval=0.2)
    q1 = jax.vmap(wrap_to_pi)(q1)
    return jnp.stack([q1, q2, p1, p2], axis=1)


# ============================================================
# 10. Visualization
# ============================================================

def plot_training(history_cost, history_kp, history_kv, result_dir):
    result_dir = Path(result_dir)
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(history_cost, marker="o", markersize=3)
    axs[0].set_ylabel("mean cost")
    axs[0].grid(True)
    axs[0].set_title("DEER policy-iteration training")

    axs[1].plot(history_kp, marker="o", markersize=3)
    axs[1].set_ylabel("kp")
    axs[1].grid(True)

    axs[2].plot(history_kv, marker="o", markersize=3)
    axs[2].set_ylabel("kv")
    axs[2].set_xlabel("iteration")
    axs[2].grid(True)

    plt.tight_layout()
    path = result_dir / "training_history.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_rollout(xs, us, costs, phys: IWPPhysicalParams, result_dir):
    result_dir = Path(result_dir)
    xs = np.asarray(xs)
    us = np.asarray(us)
    costs = np.asarray(costs)

    t_x = np.arange(xs.shape[0]) * phys.dt
    t_u = np.arange(us.shape[0]) * phys.dt

    q1 = np.arctan2(np.sin(xs[:, 0]), np.cos(xs[:, 0]))
    q2 = xs[:, 1]
    p1 = xs[:, 2]
    p2 = xs[:, 3]

    fig, axs = plt.subplots(6, 1, figsize=(9, 10), sharex=False)

    axs[0].plot(t_x, q1)
    axs[0].axhline(0.0, linestyle="--")
    axs[0].set_ylabel("q1")
    axs[0].grid(True)

    axs[1].plot(t_x, q2)
    axs[1].axhline(0.0, linestyle="--")
    axs[1].set_ylabel("q2")
    axs[1].grid(True)

    axs[2].plot(t_x, p1)
    axs[2].axhline(0.0, linestyle="--")
    axs[2].set_ylabel("p1")
    axs[2].grid(True)

    axs[3].plot(t_x, p2)
    axs[3].axhline(0.0, linestyle="--")
    axs[3].set_ylabel("p2")
    axs[3].grid(True)

    axs[4].plot(t_u, us)
    axs[4].axhline(phys.u_max, linestyle="--")
    axs[4].axhline(-phys.u_max, linestyle="--")
    axs[4].set_ylabel("u")
    axs[4].grid(True)

    axs[5].plot(t_u, costs)
    axs[5].set_ylabel("cost")
    axs[5].set_xlabel("time [s]")
    axs[5].grid(True)

    plt.suptitle("Inertia-wheel pendulum final rollout")
    plt.tight_layout()
    path = result_dir / "final_rollout_timeseries.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_iwp_gif(xs, phys: IWPPhysicalParams, filename, fps=30, frame_skip=2):
    """Visualize pendulum link plus internal wheel. q1=0 is upright."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    xs = np.asarray(xs)
    q1 = np.arctan2(np.sin(xs[:, 0]), np.cos(xs[:, 0]))
    q2 = xs[:, 1]

    q1_frames = q1[::frame_skip]
    q2_frames = q2[::frame_skip]

    link_length = 1.0
    wheel_radius = 0.18

    fig, ax = plt.subplots(figsize=(5, 5))
    limit = 1.35 * link_length
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(True)

    ax.plot([0, 0], [0, link_length], linestyle="--", linewidth=1)

    link_line, = ax.plot([], [], linewidth=3)
    wheel_circle = plt.Circle((0, 0), wheel_radius, fill=False, linewidth=2)
    ax.add_patch(wheel_circle)
    spoke_line, = ax.plot([], [], linewidth=2)
    hub_point, = ax.plot([], [], "o", markersize=5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def init():
        link_line.set_data([], [])
        spoke_line.set_data([], [])
        hub_point.set_data([], [])
        wheel_circle.center = (0, 0)
        time_text.set_text("")
        return link_line, spoke_line, hub_point, wheel_circle, time_text

    def update(frame):
        th = q1_frames[frame]
        wh = q2_frames[frame]

        px = link_length * np.sin(th)
        py = link_length * np.cos(th)

        link_line.set_data([0, px], [0, py])
        wheel_circle.center = (px, py)

        spoke_angle = th + wh
        sx = px + wheel_radius * np.sin(spoke_angle)
        sy = py + wheel_radius * np.cos(spoke_angle)

        spoke_line.set_data([px, sx], [py, sy])
        hub_point.set_data([px], [py])

        t = frame * frame_skip * phys.dt
        time_text.set_text(f"t = {t:.2f} s")
        return link_line, spoke_line, hub_point, wheel_circle, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(q1_frames),
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return filename


# ============================================================
# 11. Main
# ============================================================

def main():
    phys = IWPPhysicalParams(m11=1.0, m22=0.20, m3=1.0, dt=0.02, u_max=5.0)
    ida = IDAStructureParams(gamma1=1.50, gamma2=1.00, k3=1.00, k4=1.00)
    weights = CostWeights()

    key = jr.PRNGKey(SEED)
    key_samples, key_train = jr.split(key)

    # Initial stabilizing gains for bounded IDA-PBC controller.
    raw_gains = make_raw_gains_from_kp_kv(kp=0.60, kv=0.60, phys=phys, ida=ida)
    adam_state = adam_init(raw_gains)
    x0_batch = sample_initial_conditions(key_samples, BATCH_SIZE)

    history_cost = []
    history_kp = []
    history_kv = []

    print("Starting IDA-PBC parameter optimization with DEER")
    print("State x = [q1, q2, p1, p2]")
    print(f"T_HORIZON={T_HORIZON}, BATCH_SIZE={BATCH_SIZE}")
    print(f"u_max={phys.u_max}, gamma1={ida.gamma1}")
    print("Optimizing kp, kv under gamma1 + kp + kv <= u_max")

    start_time = time.perf_counter()

    for it in range(NUM_POLICY_ITERS):
        iter_key = jr.fold_in(key_train, it)

        grad, mean_cost = monte_carlo_deer_gradient(
            raw_gains,
            x0_batch,
            iter_key,
            phys,
            ida,
            weights,
        )

        raw_gains, adam_state = adam_update(raw_gains, grad, adam_state, lr=LEARNING_RATE)
        kp, kv = raw_to_kp_kv(raw_gains, phys, ida)

        history_cost.append(float(mean_cost))
        history_kp.append(float(kp))
        history_kv.append(float(kv))

        if it == 0 or (it + 1) % 5 == 0 or it + 1 == NUM_POLICY_ITERS:
            grad_norm = float(jnp.linalg.norm(grad))
            print(
                f"iter={it:03d} | "
                f"cost={float(mean_cost):.6e} | "
                f"kp={float(kp):.4f} | "
                f"kv={float(kv):.4f} | "
                f"grad_norm={grad_norm:.3e}"
            )

    elapsed = time.perf_counter() - start_time
    kp_final, kv_final = raw_to_kp_kv(raw_gains, phys, ida)

    print("\nDone.")
    print(f"Training time: {elapsed:.3f} s")
    print(f"Final kp={float(kp_final):.6f}")
    print(f"Final kv={float(kv_final):.6f}")
    print(f"Bound check gamma1+kp+kv={float(ida.gamma1 + kp_final + kv_final):.6f}")

    # Final evaluation rollout.
    x0_eval = jnp.array([jnp.pi - 0.25, 0.0, 0.0, 0.0])
    x0_eval = x0_eval.at[0].set(wrap_to_pi(x0_eval[0]))

    xs, us, costs = rollout_closed_loop(x0_eval, raw_gains, phys, ida, weights)

    np.savez(
        RESULT_DIR / "iwp_ida_pbc_deer_results.npz",
        raw_gains=np.asarray(raw_gains),
        kp=float(kp_final),
        kv=float(kv_final),
        xs=np.asarray(xs),
        us=np.asarray(us),
        costs=np.asarray(costs),
        history_cost=np.asarray(history_cost),
        history_kp=np.asarray(history_kp),
        history_kv=np.asarray(history_kv),
        m11=phys.m11,
        m22=phys.m22,
        m3=phys.m3,
        dt=phys.dt,
        u_max=phys.u_max,
        gamma1=ida.gamma1,
        gamma2=ida.gamma2,
        k3=ida.k3,
        k4=ida.k4,
    )

    training_path = plot_training(history_cost, history_kp, history_kv, RESULT_DIR)
    rollout_path = plot_rollout(xs, us, costs, phys, RESULT_DIR)
    gif_path = save_iwp_gif(xs, phys, RESULT_DIR / "iwp_ida_pbc_final_rollout.gif", fps=30, frame_skip=2)

    print("\nSaved:")
    print(" ", RESULT_DIR / "iwp_ida_pbc_deer_results.npz")
    print(" ", training_path)
    print(" ", rollout_path)
    print(" ", gif_path)


if __name__ == "__main__":
    main()