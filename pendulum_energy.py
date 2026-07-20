"""
jax_inverted_pendulum_energy_shaping.py

Differentiable JAX inverted pendulum model with:
    - massless rod
    - point mass m at the end
    - length l
    - inertia I about the pivot
    - torque input u
    - damping b
    - energy-shaping controller example
    - trajectory plotting
    - GIF visualization

Coordinate convention:
    theta = 0      means upright
    theta = pi     means hanging downward
    omega = theta_dot

Dynamics:
    I theta_ddot = m g l sin(theta) + u - b theta_dot

Energy relative to upright:
    E(theta, omega) = 0.5 I omega^2 + m g l (cos(theta) - 1)

At the upright equilibrium:
    theta = 0, omega = 0, E = 0

At the downward equilibrium:
    theta = pi, omega = 0, E = -2 m g l
"""

from pathlib import Path
from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class PendulumParams(NamedTuple):
    m: float = 1.0
    l: float = 1.0
    I: float = 1.0
    g: float = 9.81
    b: float = 0.05
    dt: float = 0.02
    u_max: float = 5.0


class EnergyShapingGains(NamedTuple):
    k_energy: float = 2.0
    k_p: float = 30.0
    k_d: float = 6.0
    switch_angle: float = 0.35
    switch_omega: float = 3.0


def point_mass_inertia(m, l):
    """Massless rod with point mass at end: I = m l^2."""
    return m * l**2


def wrap_to_pi(theta):
    return jnp.arctan2(jnp.sin(theta), jnp.cos(theta))


def mechanical_energy(x, params: PendulumParams):
    """
    Energy relative to upright:
        E = 0.5 I omega^2 + m g l (cos(theta) - 1)
    """
    theta = wrap_to_pi(x[0])
    omega = x[1]
    kinetic = 0.5 * params.I * omega**2
    potential = params.m * params.g * params.l * (jnp.cos(theta))
    return kinetic + potential


def desired_energy(params: PendulumParams):
    return  params.m * params.g * params.l 


def continuous_dynamics(x, u, params: PendulumParams):
    """
    x = [theta, omega]
    I theta_ddot = m g l sin(theta) + u - b omega
    """
    theta = wrap_to_pi(x[0])
    omega = x[1]
    u = jnp.clip(u, -params.u_max, params.u_max)

    theta_dot = omega
    omega_dot = (
        params.m * params.g * params.l * jnp.sin(theta)
        + u
        - params.b * omega
    ) / params.I

    return jnp.array([theta_dot, omega_dot])


def rk4_step(x, u, params: PendulumParams):
    dt = params.dt
    k1 = continuous_dynamics(x, u, params)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, params)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, params)
    k4 = continuous_dynamics(x + dt * k3, u, params)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return jnp.array([wrap_to_pi(x_next[0]), x_next[1]])


def energy_shaping_controller(x, params: PendulumParams, gains: EnergyShapingGains):
    """
    Swing-up controller:
        u_E = -k_E (E - E_des) omega

    Near upright, switch to local PD:
        u_PD = -k_p sin(theta) - k_d omega
    """
    theta = wrap_to_pi(x[0])
    omega = x[1]

    E = mechanical_energy(x, params)
    E_des = desired_energy(params)

    u_energy = -gains.k_energy * (E - E_des) * omega
    u_pd = -gains.k_p * jnp.sin(theta) - gains.k_d * omega

    # use_pd = (jnp.abs(theta) < gains.switch_angle) & (
    #     jnp.abs(omega) < gains.switch_omega
    # )

    # u = jnp.where(use_pd, u_pd, u_energy)
    u = u_energy + u_pd
    u = -2*params.m * params.g * params.l * jnp.sin(theta)-2*omega
    return jnp.clip(u, -params.u_max, params.u_max)


def simulate_closed_loop(x0, params: PendulumParams, gains: EnergyShapingGains, horizon):
    def step(x, _):
        u = energy_shaping_controller(x, params, gains)
        E = mechanical_energy(x, params)
        x_next = rk4_step(x, u, params)
        return x_next, (x, u, E)

    x_T, (xs, us, energies) = jax.lax.scan(
        step,
        x0,
        xs=None,
        length=horizon,
    )

    xs_full = jnp.vstack([xs, x_T[None, :]])
    return xs_full, us, energies


simulate_closed_loop_jit = jax.jit(simulate_closed_loop, static_argnums=(3,))


def plot_trajectory(xs, us, energies, params: PendulumParams, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    xs = np.asarray(xs)
    us = np.asarray(us)
    energies = np.asarray(energies)

    theta = np.arctan2(np.sin(xs[:, 0]), np.cos(xs[:, 0]))
    omega = xs[:, 1]

    time_state = np.arange(xs.shape[0]) * params.dt
    time_control = np.arange(us.shape[0]) * params.dt

    fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=False)

    axs[0].plot(time_state, theta)
    axs[0].axhline(0.0, linestyle="--")
    axs[0].set_ylabel(r"$\theta$ rad")
    axs[0].set_title("Inverted pendulum under energy-shaping control")
    axs[0].grid(True)

    axs[1].plot(time_state, omega)
    axs[1].axhline(0.0, linestyle="--")
    axs[1].set_ylabel(r"$\dot{\theta}$ rad/s")
    axs[1].grid(True)

    axs[2].plot(time_control, us)
    axs[2].axhline(params.u_max, linestyle="--")
    axs[2].axhline(-params.u_max, linestyle="--")
    axs[2].set_ylabel("torque u")
    axs[2].grid(True)

    axs[3].plot(time_control, energies)
    axs[3].axhline(0.0, linestyle="--", label="desired upright energy")
    axs[3].set_ylabel("energy")
    axs[3].set_xlabel("time [s]")
    axs[3].grid(True)
    axs[3].legend()

    plt.tight_layout()
    path = save_dir / "pendulum_energy_shaping_timeseries.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)

    return path


def save_pendulum_gif(xs, params: PendulumParams, filename, fps=30, frame_skip=2):
    """
    GIF of the pendulum motion.

    theta = 0 upright:
        bob = [l sin(theta), l cos(theta)]
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    xs = np.asarray(xs)
    theta = np.arctan2(np.sin(xs[:, 0]), np.cos(xs[:, 0]))
    theta_frames = theta[::frame_skip]

    fig, ax = plt.subplots(figsize=(5, 5))

    limit = 1.25 * params.l
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.grid(True)

    ax.plot([0, 0], [0, params.l], linestyle="--", linewidth=1)

    rod_line, = ax.plot([], [], linewidth=3)
    bob_point, = ax.plot([], [], "o", markersize=16)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def init():
        rod_line.set_data([], [])
        bob_point.set_data([], [])
        time_text.set_text("")
        return rod_line, bob_point, time_text

    def update(frame_index):
        th = theta_frames[frame_index]
        px = params.l * np.sin(th)
        py = params.l * np.cos(th)

        rod_line.set_data([0.0, px], [0.0, py])
        bob_point.set_data([px], [py])

        t = frame_index * frame_skip * params.dt
        time_text.set_text(f"t = {t:.2f} s")

        return rod_line, bob_point, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(theta_frames),
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return filename


def save_snapshots(xs, params: PendulumParams, filename, num_snapshots=12):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    xs = np.asarray(xs)
    theta = np.arctan2(np.sin(xs[:, 0]), np.cos(xs[:, 0]))
    indices = np.linspace(0, len(theta) - 1, num_snapshots, dtype=int)

    fig, axes = plt.subplots(1, num_snapshots, figsize=(1.1 * num_snapshots, 2.2))

    if num_snapshots == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        th = theta[idx]

        px = params.l * np.sin(th)
        py = params.l * np.cos(th)

        ax.plot([0.0, px], [0.0, py], linewidth=2)
        ax.plot([px], [py], "o", markersize=8)
        ax.plot([0, 0], [0, params.l], linestyle="--", linewidth=0.7)

        ax.set_xlim(-1.2 * params.l, 1.2 * params.l)
        ax.set_ylim(-1.2 * params.l, 1.2 * params.l)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"t={idx * params.dt:.1f}s", fontsize=8)

    plt.suptitle("Pendulum motion snapshots")
    plt.tight_layout()

    plt.savefig(filename, dpi=200)
    plt.close(fig)

    return filename


def main():
    result_dir = Path("jax_inverted_pendulum_energy_shaping_results")
    result_dir.mkdir(parents=True, exist_ok=True)

    m = 1.0
    l = 1.0

    params = PendulumParams(
        m=m,
        l=l,
        I=point_mass_inertia(m, l),
        g=9.81,
        b=0.05,
        dt=0.02,
        u_max=50.0,
    )

    gains = EnergyShapingGains(
        k_energy=2.0,
        k_p=35.0,
        k_d=8.0,
        switch_angle=0.35,
        switch_omega=3.0,
    )

    # Start near downward. Exactly [pi, 0] gives zero initial energy torque.
    x0 = jnp.array([jnp.pi - 0.05, 0.0])

    horizon = int(12.0 / params.dt)

    xs, us, energies = simulate_closed_loop_jit(x0, params, gains, horizon)

    np.savez(
        result_dir / "pendulum_energy_shaping_rollout.npz",
        xs=np.asarray(xs),
        us=np.asarray(us),
        energies=np.asarray(energies),
        m=params.m,
        l=params.l,
        I=params.I,
        g=params.g,
        b=params.b,
        dt=params.dt,
        u_max=params.u_max,
    )

    plot_path = plot_trajectory(xs, us, energies, params, result_dir)

    snapshot_path = save_snapshots(
        xs,
        params,
        result_dir / "pendulum_energy_shaping_snapshots.png",
        num_snapshots=12,
    )

    gif_path = save_pendulum_gif(
        xs,
        params,
        result_dir / "pendulum_energy_shaping.gif",
        fps=30,
        frame_skip=2,
    )

    print("Saved results:")
    print(" ", result_dir / "pendulum_energy_shaping_rollout.npz")
    print(" ", plot_path)
    print(" ", snapshot_path)
    print(" ", gif_path)


if __name__ == "__main__":
    main()