import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

from deer import deer_alg

try:
    from scipy.linalg import solve_discrete_are
except ImportError:
    solve_discrete_are = None


# ============================================================
# 1. Configuration
# ============================================================

T_max = 300
state_dim = 3
control_dim = 3

# Increase this for a lower-variance Monte Carlo estimate.
# DEER is called once per sample, so runtime grows linearly.
num_samples = 64

max_newton_iters = 30
tol = 1e-7
learning_rate = 1e-4

A = jnp.array([
    [-1.0, 0.0, 0.0],
    [0.0, 0.7, 0.0],
    [0.0, 0.0, -0.2],
])

B = jnp.array([
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1],
])

Q = jnp.eye(state_dim)
R = jnp.eye(control_dim)
Q_terminal = Q

# Nonzero tracking reference used by the DEER gradient experiment.
x_ref = jnp.ones(state_dim)

# Independent uniform initial-state distribution.
x0_low = 0.5 * jnp.ones(state_dim)
x0_high = 1.5 * jnp.ones(state_dim)

master_key = jr.PRNGKey(0)
key_K, key_samples, key_guesses = jr.split(master_key, 3)

# Initial constant feedback policy u = -Kx.
K = jr.normal(
    key_K,
    shape=(control_dim, state_dim),
) * 0.1


# ============================================================
# 2. Dynamics and cost functions
# ============================================================

def closed_loop_matrix(K_matrix):
    return A - B @ K_matrix


def stage_cost(x, K_matrix):
    """l(x,K) = (x-x_ref)'Q(x-x_ref) + u'Ru, u=-Kx."""
    u = -K_matrix @ x
    return (
        (x - x_ref).T @ Q @ (x - x_ref)
        + u.T @ R @ u
    )


def grad_stage_cost_x(x, K_matrix):
    """Gradient of l(x,K) with respect to x."""
    u = -K_matrix @ x
    return (
        2.0 * Q @ (x - x_ref)
        - 2.0 * K_matrix.T @ R @ u
    )


def direct_stage_gradient_K(x, K_matrix):
    """Direct partial derivative of l(x,K) with respect to K."""
    u = -K_matrix @ x
    return -jnp.outer(2.0 * R @ u, x)


# ============================================================
# 3. Forward rollout
# ============================================================

def rollout_state(x0, K_matrix):
    """
    Return:
        x_traj = [x_0, ..., x_{T-1}], shape (T_max, n)
        x_T    = terminal state
    """
    A_cl = closed_loop_matrix(K_matrix)

    def forward_step(x_k, _):
        x_next = A_cl @ x_k
        return x_next, x_k

    x_T, x_traj = jax.lax.scan(
        forward_step,
        x0,
        xs=None,
        length=T_max,
    )

    return x_traj, x_T


# ============================================================
# 4. Robust DEER-result parsing
# ============================================================

def parse_deer_result(deer_result):
    """
    Different deer package versions may not return an iteration count.
    This helper always returns z_deer and returns None when the count is
    unavailable.
    """
    if not isinstance(deer_result, tuple):
        raise TypeError(
            "deer_alg was expected to return a tuple, but returned "
            f"{type(deer_result)}."
        )

    if len(deer_result) < 2:
        raise ValueError(
            "deer_alg returned fewer than two values; cannot extract "
            "the trajectory."
        )

    z_deer = deer_result[1]
    newton_steps = deer_result[2] if len(deer_result) > 2 else None
    return z_deer, newton_steps


# ============================================================
# 5. DEER gradient for one sampled initial condition
# ============================================================

def deer_gradient_single(x0, K_matrix, guess_key):
    """
    Estimate the finite-horizon gradient for

        J_T(x0,K) = sum_{k=0}^{T} l(x_k,K),

    with terminal condition

        lambda_T = grad_x l(x_T,K).

    The stacked DEER variable is

        z_i = [x_i, lambda_{T-i}].
    """
    A_cl = closed_loop_matrix(K_matrix)

    # x_traj = [x_0, ..., x_{T-1}]
    x_traj, x_T = rollout_state(x0, K_matrix)

    # Costate driver = [x_{T-1}, ..., x_0].
    reversed_x_traj = jnp.flip(x_traj, axis=0)

    def stacked_f(z, driver_x):
        x_i = z[:state_dim]
        lambda_T_minus_i = z[state_dim:]

        # Forward state dynamics.
        x_next = A_cl @ x_i

        # Backward costate dynamics.
        lambda_previous = (
            grad_stage_cost_x(driver_x, K_matrix)
            + A_cl.T @ lambda_T_minus_i
        )

        return jnp.concatenate([x_next, lambda_previous])

    # Terminal condition for the objective sum_{k=0}^{T} l(x_k,K).
    lambda_T = grad_stage_cost_x(x_T, K_matrix)

    # The initial stacked boundary contains x_0 and lambda_T.
    z0 = jnp.concatenate([x0, lambda_T])

    # Guess for z_1, ..., z_T.
    z_guess = 0.1 * jr.normal(
        guess_key,
        shape=(T_max, 2 * state_dim),
    )

    deer_result = deer_alg(
        stacked_f,
        z0,
        z_guess,
        reversed_x_traj,
        num_iters=max_newton_iters,
        full_trace=True,
        Ts=None,
        tol=tol,
    )

    z_deer, newton_steps = parse_deer_result(deer_result)

    # z_deer state block = [x_1, ..., x_T].
    x_deer = z_deer[:, :state_dim]

    # z_deer costate block = [lambda_{T-1}, ..., lambda_0].
    lambda_deer_reversed = z_deer[:, state_dim:]

    # Convert to [lambda_0, ..., lambda_{T-1}].
    lambda_chronological = jnp.flip(
        lambda_deer_reversed,
        axis=0,
    )

    # Required gradient pairing = [lambda_1, ..., lambda_T].
    lambda_k_plus_1 = jnp.vstack([
        lambda_chronological[1:],
        lambda_T[None, :],
    ])

    def gradient_step(x_k, lambda_next):
        u_k = -K_matrix @ x_k
        h_u = 2.0 * R @ u_k + B.T @ lambda_next
        return -jnp.outer(h_u, x_k)

    gradient_terms = jax.vmap(gradient_step)(
        x_traj,
        lambda_k_plus_1,
    )

    # k = 0, ..., T-1 contributions.
    gradient = jnp.sum(gradient_terms, axis=0)

    # Direct terminal-stage contribution d l(x_T,K) / dK.
    gradient = gradient + direct_stage_gradient_K(x_T, K_matrix)

    # DEER state trajectory consistency diagnostic.
    rollout_states_1_to_T = jnp.vstack([x_traj[1:], x_T[None, :]])
    state_error = (
        jnp.linalg.norm(x_deer - rollout_states_1_to_T)
        / jnp.maximum(
            jnp.linalg.norm(rollout_states_1_to_T),
            1e-14,
        )
    )

    return gradient, newton_steps, state_error


# ============================================================
# 6. Uniform-distribution moments
# ============================================================

def uniform_initial_moments(low, high):
    """Return mu_0=E[x_0] and S_0=E[x_0 x_0^T]."""
    mu_0 = 0.5 * (low + high)
    variances = ((high - low) ** 2) / 12.0
    covariance_0 = jnp.diag(variances)
    S_0 = covariance_0 + jnp.outer(mu_0, mu_0)
    return mu_0, S_0


# ============================================================
# 7. Exact expected gradient for the same constant-K objective
# ============================================================

def exact_expected_gradient_uniform(K_matrix, low, high):
    """
    Exact gradient of

        E[sum_{k=0}^{T} l(x_k,K)]

    for independent uniform initial-state components.

    The costate is affine:
        lambda_k = P_k x_k + p_k.
    """
    A_cl = closed_loop_matrix(K_matrix)
    mu_k, S_k = uniform_initial_moments(low, high)

    H = Q + K_matrix.T @ R @ K_matrix

    # Terminal condition lambda_T = grad_x l(x_T,K).
    P = 2.0 * H
    p = -2.0 * Q @ x_ref

    P_sequence = [None] * (T_max + 1)
    p_sequence = [None] * (T_max + 1)
    P_sequence[T_max] = P
    p_sequence[T_max] = p

    # Backward affine costate recursion.
    for k in range(T_max - 1, -1, -1):
        P = 2.0 * H + A_cl.T @ P @ A_cl
        p = -2.0 * Q @ x_ref + A_cl.T @ p
        P_sequence[k] = P
        p_sequence[k] = p

    expected_gradient = jnp.zeros_like(K_matrix)

    # k = 0, ..., T-1.
    for k in range(T_max):
        P_next = P_sequence[k + 1]
        p_next = p_sequence[k + 1]

        # E[lambda_{k+1} x_k^T].
        expected_lambda_x = (
            P_next @ A_cl @ S_k
            + jnp.outer(p_next, mu_k)
        )

        expected_gradient = (
            expected_gradient
            + 2.0 * R @ K_matrix @ S_k
            - B.T @ expected_lambda_x
        )

        mu_k = A_cl @ mu_k
        S_k = A_cl @ S_k @ A_cl.T

    # Terminal direct derivative, using S_k = E[x_T x_T^T].
    expected_gradient = (
        expected_gradient
        + 2.0 * R @ K_matrix @ S_k
    )

    return expected_gradient


# ============================================================
# 8. Exact expected cost and autodiff verification
# ============================================================

def exact_expected_cost_uniform(K_matrix, low, high):
    """Exact E[sum_{k=0}^{T} l(x_k,K)]."""
    A_cl = closed_loop_matrix(K_matrix)
    mu_k, S_k = uniform_initial_moments(low, high)

    H = Q + K_matrix.T @ R @ K_matrix
    reference_constant = x_ref.T @ Q @ x_ref

    expected_cost = 0.0

    for k in range(T_max + 1):
        expected_stage_cost = (
            jnp.trace(H @ S_k)
            - 2.0 * x_ref.T @ Q @ mu_k
            + reference_constant
        )
        expected_cost = expected_cost + expected_stage_cost

        if k < T_max:
            mu_k = A_cl @ mu_k
            S_k = A_cl @ S_k @ A_cl.T

    return expected_cost


# ============================================================
# 9. Analytic finite-horizon LQT benchmark
# ============================================================

def finite_horizon_lqt_gains(
    A_matrix,
    B_matrix,
    Q_matrix,
    R_matrix,
    Q_terminal_matrix,
    reference,
    horizon,
):
    """
    Exact finite-horizon LQT solution for the standard objective

        sum_{k=0}^{T-1} [(x_k-r)'Q(x_k-r) + u_k'Ru_k]
        + (x_T-r)'Q_f(x_T-r).

    The optimal policy is time-varying and affine:

        u_k = -K_k x_k + d_k.

    This is an analytic benchmark. It is not the same constrained
    problem as using one constant K at every time and including the
    terminal control cost u_T'Ru_T.
    """
    P_next = Q_terminal_matrix
    s_next = -Q_terminal_matrix @ reference

    K_sequence = [None] * horizon
    d_sequence = [None] * horizon
    P_sequence = [None] * (horizon + 1)
    s_sequence = [None] * (horizon + 1)

    P_sequence[horizon] = P_next
    s_sequence[horizon] = s_next

    for k in range(horizon - 1, -1, -1):
        control_hessian = (
            R_matrix
            + B_matrix.T @ P_next @ B_matrix
        )

        K_k = jnp.linalg.solve(
            control_hessian,
            B_matrix.T @ P_next @ A_matrix,
        )

        d_k = -jnp.linalg.solve(
            control_hessian,
            B_matrix.T @ s_next,
        )

        A_cl_k = A_matrix - B_matrix @ K_k

        P_k = (
            Q_matrix
            + A_matrix.T @ P_next @ A_matrix
            - A_matrix.T
            @ P_next
            @ B_matrix
            @ jnp.linalg.solve(
                control_hessian,
                B_matrix.T @ P_next @ A_matrix,
            )
        )

        s_k = -Q_matrix @ reference + A_cl_k.T @ s_next

        K_sequence[k] = K_k
        d_sequence[k] = d_k
        P_sequence[k] = P_k
        s_sequence[k] = s_k

        P_next = P_k
        s_next = s_k

    return (
        jnp.stack(K_sequence),
        jnp.stack(d_sequence),
        jnp.stack(P_sequence),
        jnp.stack(s_sequence),
    )


def expected_standard_lqt_cost(
    K_sequence,
    d_sequence,
    low,
    high,
):
    """
    Expected cost of the standard finite-horizon affine LQT policy.
    Used only to evaluate the analytic LQT benchmark.
    """
    mu_k, S_k = uniform_initial_moments(low, high)
    expected_cost = 0.0
    reference_constant = x_ref.T @ Q @ x_ref

    for k in range(T_max):
        K_k = K_sequence[k]
        d_k = d_sequence[k]

        # E[u_k u_k^T] for u_k = -K_k x_k + d_k.
        expected_uu = (
            K_k @ S_k @ K_k.T
            - K_k @ jnp.outer(mu_k, d_k)
            - jnp.outer(d_k, mu_k) @ K_k.T
            + jnp.outer(d_k, d_k)
        )

        expected_cost = expected_cost + (
            jnp.trace(Q @ S_k)
            - 2.0 * x_ref.T @ Q @ mu_k
            + reference_constant
            + jnp.trace(R @ expected_uu)
        )

        A_cl_k = A - B @ K_k
        affine_term = B @ d_k

        S_k = (
            A_cl_k @ S_k @ A_cl_k.T
            + A_cl_k @ jnp.outer(mu_k, affine_term)
            + jnp.outer(affine_term, mu_k) @ A_cl_k.T
            + jnp.outer(affine_term, affine_term)
        )
        mu_k = A_cl_k @ mu_k + affine_term

    # Standard terminal state cost only.
    expected_cost = expected_cost + (
        jnp.trace(Q_terminal @ S_k)
        - 2.0 * x_ref.T @ Q_terminal @ mu_k
        + x_ref.T @ Q_terminal @ x_ref
    )

    return expected_cost


def rollout_time_varying_lqt(x0, K_sequence, d_sequence):
    """Roll out the analytic time-varying affine LQT policy."""
    states = [x0]
    controls = []
    x_k = x0

    for k in range(T_max):
        u_k = -K_sequence[k] @ x_k + d_sequence[k]
        x_k = A @ x_k + B @ u_k
        controls.append(u_k)
        states.append(x_k)

    return jnp.stack(states), jnp.stack(controls)


# ============================================================
# 10. Sample initial states and run DEER Monte Carlo
# ============================================================

x0_samples = jr.uniform(
    key_samples,
    shape=(num_samples, state_dim),
    minval=x0_low,
    maxval=x0_high,
)

guess_keys = jr.split(key_guesses, num_samples)

deer_sample_gradients = []
deer_newton_steps = []
deer_state_errors = []

print(f"Running DEER for {num_samples} sampled initial states...")

for sample_index in range(num_samples):
    gradient_i, steps_i, state_error_i = deer_gradient_single(
        x0_samples[sample_index],
        K,
        guess_keys[sample_index],
    )

    deer_sample_gradients.append(gradient_i)
    deer_newton_steps.append(steps_i)
    deer_state_errors.append(state_error_i)

    if (
        sample_index == 0
        or (sample_index + 1) % max(1, num_samples // 8) == 0
        or sample_index + 1 == num_samples
    ):
        print(f"  Completed {sample_index + 1}/{num_samples}")

deer_sample_gradients = jnp.stack(deer_sample_gradients, axis=0)
deer_state_errors = jnp.asarray(deer_state_errors)

gradient_deer_mc = jnp.mean(deer_sample_gradients, axis=0)
gradient_deer_std = jnp.std(
    deer_sample_gradients,
    axis=0,
    ddof=1,
)
gradient_deer_standard_error = (
    gradient_deer_std / jnp.sqrt(num_samples)
)


# ============================================================
# 11. Exact-gradient comparison
# ============================================================

gradient_analytic = exact_expected_gradient_uniform(
    K,
    x0_low,
    x0_high,
)

gradient_analytic_autodiff = jax.grad(
    exact_expected_cost_uniform,
    argnums=0,
)(
    K,
    x0_low,
    x0_high,
)

gradient_error = gradient_deer_mc - gradient_analytic
absolute_error = jnp.linalg.norm(gradient_error, ord="fro")
relative_error = (
    absolute_error
    / jnp.maximum(
        jnp.linalg.norm(gradient_analytic, ord="fro"),
        1e-14,
    )
)

analytic_verification_error = jnp.linalg.norm(
    gradient_analytic - gradient_analytic_autodiff,
    ord="fro",
)


# ============================================================
# 12. Analytic LQT and DARE benchmarks
# ============================================================

(
    K_lqt_sequence,
    d_lqt_sequence,
    P_lqt_sequence,
    s_lqt_sequence,
) = finite_horizon_lqt_gains(
    A,
    B,
    Q,
    R,
    Q_terminal,
    x_ref,
    T_max,
)

expected_lqt_cost = expected_standard_lqt_cost(
    K_lqt_sequence,
    d_lqt_sequence,
    x0_low,
    x0_high,
)

K_dare = None
if solve_discrete_are is not None:
    try:
        P_dare = solve_discrete_are(
            np.asarray(A),
            np.asarray(B),
            np.asarray(Q),
            np.asarray(R),
        )

        K_dare = np.linalg.solve(
            np.asarray(R)
            + np.asarray(B).T @ P_dare @ np.asarray(B),
            np.asarray(B).T @ P_dare @ np.asarray(A),
        )
    except Exception as exc:
        print("\nDARE benchmark could not be computed:")
        print(exc)


# ============================================================
# 13. Diagnostics and printed results
# ============================================================

A_cl = closed_loop_matrix(K)
closed_loop_eigenvalues = jnp.linalg.eigvals(A_cl)
spectral_radius = jnp.max(jnp.abs(closed_loop_eigenvalues))

print("\nClosed-loop eigenvalues for the current constant K:")
print(closed_loop_eigenvalues)

print("\nClosed-loop spectral radius:")
print(spectral_radius)

if spectral_radius >= 1.0:
    print(
        "\nWARNING: The current K is not asymptotically stabilizing. "
        "The finite-horizon calculation remains defined, but state "
        "and gradient magnitudes may grow with T."
    )

print("\nDEER Monte Carlo expected gradient:")
print(gradient_deer_mc)

print("\nExact analytic expected gradient:")
print(gradient_analytic)

print("\nAutodiff gradient of the exact expected cost:")
print(gradient_analytic_autodiff)

print("\nMonte Carlo componentwise standard error:")
print(gradient_deer_standard_error)

print("\nAbsolute Frobenius gradient error:")
print(absolute_error)

print("\nRelative Frobenius gradient error:")
print(relative_error)

print("\nAnalytic recursion vs. autodiff error:")
print(analytic_verification_error)

# Handle deer versions that return None for the iteration count.
valid_newton_steps = []
for steps in deer_newton_steps:
    if steps is None:
        continue

    try:
        steps_array = np.asarray(steps)
        if steps_array.size == 1:
            valid_newton_steps.append(float(steps_array.reshape(-1)[0]))
    except (TypeError, ValueError):
        pass

print("\nMean DEER Newton steps:")
if valid_newton_steps:
    print(float(np.mean(valid_newton_steps)))
else:
    print(
        "Not available: this deer_alg version returned None or a "
        "non-scalar object for the iteration-count output."
    )

print("\nMaximum DEER state-rollout relative error:")
print(jnp.max(deer_state_errors))

print("\nAnalytic finite-horizon LQT gain K_0:")
print(K_lqt_sequence[0])

print("\nAnalytic finite-horizon LQT feedforward d_0:")
print(d_lqt_sequence[0])

print("\nAnalytic finite-horizon LQT gain K_{T-1}:")
print(K_lqt_sequence[-1])

print("\nExpected cost of the analytic standard LQT policy:")
print(expected_lqt_cost)

if K_dare is not None:
    print("\nInfinite-horizon zero-reference DARE gain:")
    print(K_dare)

    print("\nFrobenius distance from current K to DARE gain:")
    print(np.linalg.norm(np.asarray(K) - K_dare, ord="fro"))

print(
    "\nNOTE: The finite-horizon LQT gains are the exact analytic "
    "solution of the standard time-varying affine tracking problem. "
    "They are not the exact optimizer of the constrained problem in "
    "which one constant K is shared across all time steps and the "
    "terminal control cost is included."
)


# ============================================================
# 14. Policy update using the DEER Monte Carlo gradient
# ============================================================

K_new = K - learning_rate * gradient_deer_mc

print("\nInitial constant feedback matrix K:")
print(K)

print("\nUpdated constant feedback matrix K_new:")
print(K_new)


# ============================================================
# 15. Monte Carlo convergence data
# ============================================================

cumulative_gradient = jnp.cumsum(deer_sample_gradients, axis=0)

sample_counts = np.unique(
    np.geomspace(
        1,
        num_samples,
        num=min(15, num_samples),
    ).astype(int)
)

relative_errors_by_sample_count = []

for count in sample_counts:
    gradient_at_count = cumulative_gradient[count - 1] / count
    error_at_count = (
        jnp.linalg.norm(
            gradient_at_count - gradient_analytic,
            ord="fro",
        )
        / jnp.maximum(
            jnp.linalg.norm(gradient_analytic, ord="fro"),
            1e-14,
        )
    )
    relative_errors_by_sample_count.append(float(error_at_count))


# ============================================================
# 16. Visualization: Monte Carlo convergence
# ============================================================

plt.figure(figsize=(7, 5))
plt.loglog(
    sample_counts,
    relative_errors_by_sample_count,
    marker="o",
    label="DEER Monte Carlo error",
)

reference_rate = (
    relative_errors_by_sample_count[0]
    / np.sqrt(sample_counts)
)
plt.loglog(
    sample_counts,
    reference_rate,
    linestyle="--",
    label=r"$N^{-1/2}$ reference",
)

plt.xlabel("Number of sampled initial states")
plt.ylabel("Relative gradient error")
plt.title("DEER Monte Carlo gradient convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 17. Visualization: analytic vs. DEER gradient components
# ============================================================

gradient_mc_flat = np.asarray(gradient_deer_mc).reshape(-1)
gradient_analytic_flat = np.asarray(gradient_analytic).reshape(-1)
component_indices = np.arange(gradient_mc_flat.size)
bar_width = 0.4

plt.figure(figsize=(9, 5))
plt.bar(
    component_indices - bar_width / 2,
    gradient_analytic_flat,
    width=bar_width,
    label="Analytic",
)
plt.bar(
    component_indices + bar_width / 2,
    gradient_mc_flat,
    width=bar_width,
    label="DEER Monte Carlo",
)
plt.xlabel("Flattened gradient component")
plt.ylabel("Gradient value")
plt.title("DEER Monte Carlo versus analytic gradient")
plt.xticks(component_indices)
plt.grid(True, axis="y")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 18. Visualization: absolute componentwise gradient error
# ============================================================

plt.figure(figsize=(6, 5))
plt.imshow(
    np.abs(np.asarray(gradient_error)),
    aspect="auto",
)
plt.colorbar(label="Absolute error")
plt.xticks(
    range(state_dim),
    [f"x{j + 1}" for j in range(state_dim)],
)
plt.yticks(
    range(control_dim),
    [f"u{i + 1}" for i in range(control_dim)],
)
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Absolute DEER Monte Carlo gradient error")
plt.tight_layout()
plt.show()


# ============================================================
# 19. Visualization: current K
# ============================================================

plt.figure(figsize=(6, 5))
plt.imshow(np.asarray(K), aspect="auto")
plt.colorbar(label="Gain value")
plt.xticks(
    range(state_dim),
    [f"x{j + 1}" for j in range(state_dim)],
)
plt.yticks(
    range(control_dim),
    [f"u{i + 1}" for i in range(control_dim)],
)
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Current constant gain K")
plt.tight_layout()
plt.show()


# ============================================================
# 20. Visualization: analytic finite-horizon K_0
# ============================================================

plt.figure(figsize=(6, 5))
plt.imshow(np.asarray(K_lqt_sequence[0]), aspect="auto")
plt.colorbar(label="Gain value")
plt.xticks(
    range(state_dim),
    [f"x{j + 1}" for j in range(state_dim)],
)
plt.yticks(
    range(control_dim),
    [f"u{i + 1}" for i in range(control_dim)],
)
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Analytic finite-horizon LQT gain K_0")
plt.tight_layout()
plt.show()


# ============================================================
# 21. Visualization: variation of analytic LQT gains
# ============================================================

gain_distance_from_initial = np.asarray([
    np.linalg.norm(
        np.asarray(K_lqt_sequence[k])
        - np.asarray(K_lqt_sequence[0]),
        ord="fro",
    )
    for k in range(T_max)
])

plt.figure(figsize=(7, 5))
plt.plot(
    np.arange(T_max),
    gain_distance_from_initial,
)
plt.xlabel("Time step k")
plt.ylabel(r"$\|K_k^\star-K_0^\star\|_F$")
plt.title("Time variation of the analytic finite-horizon gain")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 22. Visualization: representative state trajectories
# ============================================================

representative_x0 = 0.5 * (x0_low + x0_high)

constant_x_traj, constant_x_T = rollout_state(
    representative_x0,
    K,
)
constant_states = jnp.vstack([
    constant_x_traj,
    constant_x_T[None, :],
])

lqt_states, _ = rollout_time_varying_lqt(
    representative_x0,
    K_lqt_sequence,
    d_lqt_sequence,
)

plt.figure(figsize=(8, 5))
for state_index in range(state_dim):
    plt.plot(
        np.arange(T_max + 1),
        np.asarray(constant_states[:, state_index]),
        linestyle="--",
        label=f"Constant K: x{state_index + 1}",
    )
    plt.plot(
        np.arange(T_max + 1),
        np.asarray(lqt_states[:, state_index]),
        label=f"Analytic LQT: x{state_index + 1}",
    )

plt.xlabel("Time step k")
plt.ylabel("State value")
plt.title("Representative state trajectories")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 23. Optional DARE-gain visualization
# ============================================================

if K_dare is not None:
    plt.figure(figsize=(6, 5))
    plt.imshow(K_dare, aspect="auto")
    plt.colorbar(label="Gain value")
    plt.xticks(
        range(state_dim),
        [f"x{j + 1}" for j in range(state_dim)],
    )
    plt.yticks(
        range(control_dim),
        [f"u{i + 1}" for i in range(control_dim)],
    )
    plt.xlabel("State component")
    plt.ylabel("Control component")
    plt.title("Infinite-horizon zero-reference DARE gain")
    plt.tight_layout()
    plt.show()
