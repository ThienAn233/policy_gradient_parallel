"""JAX simulation of the inertia-wheel pendulum with IDA-PBC.

Paper notation:

    q = [q_1, q_2]^T,       p = [p_1, p_2]^T
    M = diag(m_11, m_22),   G = [-1, 1]^T

Hamiltonian dynamics:

    q_1_dot = p_1 / m_11
    q_2_dot = p_2 / m_22
    p_1_dot = m_3 sin(q_1) - u
    p_2_dot = u

IDA-PBC controller:

    u = gamma_1 sin(q_1)
        + k_p (q_2 + gamma_2 q_1)
        + k_v k_2 (q_2_dot + gamma_2 q_1_dot)

The ODE is integrated with a JAX-compatible fixed-step RK4 method. Running
this file creates ``inertia_wheel_pendulum_jax.gif``.

Dependencies:
    pip install "jax[cpu]" matplotlib pillow
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------
# Parameters and gains from the paper's inertia-wheel-pendulum example
# -----------------------------------------------------------------------------
m_11 = 0.1
m_22 = 0.2
m_3 = 10.0

a_1 = 1.0
a_2 = -1.5
a_3 = 6.0

k_p = 3.75
k_v = 10.0

gamma_1 = (a_2 / (a_1 + a_2)) * m_3
gamma_2 = -m_11 * (a_2 + a_3) / (m_22 * (a_1 + a_2))
k_2 = -m_22 * (a_1 + a_2) / (a_1 * a_3 - a_2**2)


# -----------------------------------------------------------------------------
# IDA-PBC and closed-loop dynamics
# -----------------------------------------------------------------------------
def ida_pbc(x):
    """Control input u for x = [q_1, q_2, p_1, p_2]."""
    q_1, q_2, p_1, p_2 = x

    q_1_dot = p_1 / m_11
    q_2_dot = p_2 / m_22

    u_es = gamma_1 * jnp.sin(q_1) + k_p * (q_2 + gamma_2 * q_1)
    u_di = k_v * k_2 * (q_2_dot + gamma_2 * q_1_dot)

    return u_es + u_di


def dynamics(x):
    """Closed-loop Hamiltonian dynamics."""
    q_1, q_2, p_1, p_2 = x
    u = ida_pbc(x)

    return jnp.array(
        [
            p_1 / m_11,
            p_2 / m_22,
            m_3 * jnp.sin(q_1) - u,
            u,
        ]
    )


def rk4_step(x, dt):
    """One fourth-order Runge-Kutta step."""
    k1 = dynamics(x)
    k2_rk = dynamics(x + 0.5 * dt * k1)
    k3 = dynamics(x + 0.5 * dt * k2_rk)
    k4 = dynamics(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2_rk + 2.0 * k3 + k4)


# -----------------------------------------------------------------------------
# JAX simulation
# -----------------------------------------------------------------------------
t_0 = 0.0
t_f = 10.0
dt = 0.01
num_steps = int((t_f - t_0) / dt)

# Almost downward and initially at rest, as in the paper's simulation.
x_0 = jnp.array([3.14, 0.0, 0.0, 0.0])


@jax.jit
def simulate(x_initial):
    def scan_step(x, _):
        x_next = rk4_step(x, dt)
        return x_next, x_next

    _, x_history = jax.lax.scan(scan_step, x_initial, xs=None, length=num_steps)
    return jnp.vstack((x_initial, x_history))


x = simulate(x_0)
t = jnp.linspace(t_0, t_f, num_steps + 1)
u = jax.vmap(ida_pbc)(x)

# Matplotlib expects host NumPy arrays.
t, x, u = jax.device_get((t, x, u))
q_1, q_2, p_1, p_2 = x.T


# -----------------------------------------------------------------------------
# GIF animation
# -----------------------------------------------------------------------------
rod_length = 1.0
wheel_radius = 0.18
fps = int(1.0/dt)
frame_step = 1+0*max(1, int(0.20 / dt))
frames = np.arange(0, len(t), frame_step)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.25, 1.35)
ax.grid(True)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Inertia-Wheel Pendulum: JAX IDA-PBC")

ax.plot(0.0, 0.0, "ko", markersize=8)
rod, = ax.plot([], [], linewidth=4)
wheel = Circle((0.0, 0.0), wheel_radius, fill=False, linewidth=3)
ax.add_patch(wheel)
spoke, = ax.plot([], [], linewidth=2)
text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")


def update(frame):
    i = frames[frame]

    # q_1 is measured from the upward vertical.
    wheel_x = rod_length * np.sin(q_1[i])
    wheel_y = rod_length * np.cos(q_1[i])

    rod.set_data([0.0, wheel_x], [0.0, wheel_y])
    wheel.center = (wheel_x, wheel_y)

    # q_2 is the absolute inertia-wheel angle.
    dx = wheel_radius * np.sin(q_2[i])
    dy = wheel_radius * np.cos(q_2[i])
    spoke.set_data(
        [wheel_x - dx, wheel_x + dx],
        [wheel_y - dy, wheel_y + dy],
    )

    text.set_text(
        f"t = {t[i]:5.2f} s\n"
        f"q_1 = {q_1[i]: .3f} rad\n"
        f"q_2 = {q_2[i]: .3f} rad\n"
        f"u   = {u[i]: .3f}"
    )

    return rod, wheel, spoke, text


animation = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=1000 / fps,
    blit=True,
)

output_file = "inertia_wheel_pendulum_jax.gif"
animation.save(output_file, writer=PillowWriter(fps=fps))
plt.close(fig)

print(f"gamma_1 = {gamma_1:.4f}")
print(f"gamma_2 = {gamma_2:.4f}")
print(f"k_2     = {k_2:.6f}")
print("final state [q_1, q_2, p_1, p_2] =")
print(x[-1])
print(f"saved: {output_file}")