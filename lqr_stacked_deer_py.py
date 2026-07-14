import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

from deer import deer_alg


# ============================================================
# 1. System definition
# ============================================================

T_max = 300
state_dim = 3
control_dim = 3

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

x_ref = jnp.ones(state_dim)

key = jr.PRNGKey(0)
key_K, key_samples, key_guesses = jr.split(key, 3)

K = jr.normal(
    key_K,
    shape=(control_dim, state_dim),
) * 0.1

tol = 1e-7
max_newton_iters = 30

# Number of Monte Carlo trajectories
num_samples = 128

# Independent uniform components:
# x0_j ~ Uniform(0.5, 1.5)
x0_low = 0.5 * jnp.ones(state_dim)
x0_high = 1.5 * jnp.ones(state_dim)


# ============================================================
# 2. Closed-loop dynamics and stage-cost gradient
# ============================================================

def closed_loop_matrix(K):
    return A - B @ K


def stage_cost(x, K):
    u = -K @ x

    return (
        (x - x_ref).T @ Q @ (x - x_ref)
        + u.T @ R @ u
    )


def grad_stage_cost_x(x, K):
    """
    Gradient with respect to x of

        l(x,K) = (x-x_ref)^T Q (x-x_ref)
                 + (-Kx)^T R (-Kx).
    """
    u = -K @ x

    return (
        2.0 * Q @ (x - x_ref)
        - 2.0 * K.T @ R @ u
    )


# ============================================================
# 3. Forward trajectory rollout
# ============================================================

def rollout_state(x0, K):
    """
    Returns:
        x_traj = [x_0, ..., x_{T-1}]
        x_T    = terminal state
    """
    A_cl = closed_loop_matrix(K)

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
# 4. DEER gradient for one initial state
# ============================================================

def deer_gradient_single(x0, K, guess_key):
    """
    Solve the stacked state-costate dynamics using DEER.

    The DEER sequence has the form

        z_i = [x_i, lambda_{T-i}].

    The finite-horizon objective is

        J_T = sum_{k=0}^{T-1} l(x_k,K),

    so lambda_T = 0.
    """
    A_cl = closed_loop_matrix(K)

    # Exact chronological state trajectory used as the costate driver:
    # x_0, ..., x_{T-1}
    x_traj, _ = rollout_state(x0, K)

    # Costate recursion needs:
    # x_{T-1}, x_{T-2}, ..., x_0
    reversed_x_traj = jnp.flip(x_traj, axis=0)

    def stacked_f(z, driver_x):
        """
        z = [x_i, lambda_{T-i}]

        driver_x = x_{T-i-1}
        """
        x_i = z[:state_dim]
        lambda_T_minus_i = z[state_dim:]

        # Forward state dynamics
        x_next = A_cl @ x_i

        # Backward costate dynamics
        grad_l_x = grad_stage_cost_x(driver_x, K)

        lambda_previous = (
            grad_l_x
            + A_cl.T @ lambda_T_minus_i
        )

        return jnp.concatenate([
            x_next,
            lambda_previous,
        ])

    lambda_T = jnp.zeros(state_dim)

    z0 = jnp.concatenate([
        x0,
        lambda_T,
    ])

    # Initial guess for z_1, ..., z_T
    z_guess = 0.1 * jr.normal(
        guess_key,
        shape=(T_max, 2 * state_dim),
    )

    _, z_deer, newton_steps, *_ = deer_alg(
        stacked_f,
        z0,
        z_guess,
        reversed_x_traj,
        num_iters=max_newton_iters,
        full_trace=True,
        Ts=None,
        tol=tol,
    )

    # z_deer[:, :3] contains x_1, ..., x_T.
    x_deer = z_deer[:, :state_dim]

    # z_deer[:, 3:] contains:
    # lambda_{T-1}, ..., lambda_0.
    lambda_deer_reversed = z_deer[:, state_dim:]

    # Convert to:
    # lambda_0, ..., lambda_{T-1}.
    lambda_chronological = jnp.flip(
        lambda_deer_reversed,
        axis=0,
    )

    # For the policy gradient, pair x_k with lambda_{k+1}.
    lambda_k_plus_1 = jnp.vstack([
        lambda_chronological[1:],
        lambda_T[None, :],
    ])

    def gradient_step(x_k, lambda_next):
        u_k = -K @ x_k

        h_u = (
            2.0 * R @ u_k
            + B.T @ lambda_next
        )

        # u = -Kx, so du/dK contributes -x.
        return -jnp.outer(h_u, x_k)

    gradient_terms = jax.vmap(gradient_step)(
        x_traj,
        lambda_k_plus_1,
    )

    gradient = jnp.sum(gradient_terms, axis=0)

    # Check whether DEER's state component agrees with the
    # ordinary closed-loop rollout.
    x_deer_error = (
        jnp.linalg.norm(x_deer[:-1] - x_traj[1:])
        / jnp.maximum(
            jnp.linalg.norm(x_traj[1:]),
            1e-14,
        )
    )

    return gradient, newton_steps, x_deer_error


# ============================================================
# 5. Sample initial states from a uniform distribution
# ============================================================

x0_samples = jr.uniform(
    key_samples,
    shape=(num_samples, state_dim),
    minval=x0_low,
    maxval=x0_high,
)

guess_keys = jr.split(
    key_guesses,
    num_samples,
)


# ============================================================
# 6. Run DEER for every Monte Carlo trajectory
# ============================================================

deer_sample_gradients = []
deer_newton_steps = []
deer_state_errors = []

for sample_index in range(num_samples):
    gradient_i, steps_i, state_error_i = deer_gradient_single(
        x0_samples[sample_index],
        K,
        guess_keys[sample_index],
    )

    deer_sample_gradients.append(gradient_i)
    deer_newton_steps.append(steps_i)
    deer_state_errors.append(state_error_i)

deer_sample_gradients = jnp.stack(
    deer_sample_gradients,
    axis=0,
)

deer_state_errors = jnp.asarray(deer_state_errors)

gradient_deer_mc = jnp.mean(
    deer_sample_gradients,
    axis=0,
)

gradient_deer_std = jnp.std(
    deer_sample_gradients,
    axis=0,
    ddof=1,
)

gradient_deer_standard_error = (
    gradient_deer_std
    / jnp.sqrt(num_samples)
)


# ============================================================
# 7. Exact moments of the uniform initial-state distribution
# ============================================================

def uniform_initial_moments(low, high):
    """
    Assumes independent uniform components.

    Returns:
        mu_0 = E[x_0]
        S_0  = E[x_0 x_0^T]
    """
    mu_0 = 0.5 * (low + high)

    variances = ((high - low) ** 2) / 12.0

    covariance_0 = jnp.diag(variances)

    S_0 = covariance_0 + jnp.outer(
        mu_0,
        mu_0,
    )

    return mu_0, S_0


# ============================================================
# 8. Exact expected finite-horizon gradient
# ============================================================

def exact_expected_gradient_uniform(K, low, high):
    """
    Exact expected gradient under an independent uniform
    initial-state distribution.

    The costate is affine in the state:

        lambda_k = P_k x_k + p_k.
    """
    A_cl = closed_loop_matrix(K)

    mu_k, S_k = uniform_initial_moments(
        low,
        high,
    )

    H = Q + K.T @ R @ K

    # Terminal condition:
    # lambda_T = 0
    P = jnp.zeros((state_dim, state_dim))
    p = jnp.zeros(state_dim)

    P_sequence = [None] * (T_max + 1)
    p_sequence = [None] * (T_max + 1)

    P_sequence[T_max] = P
    p_sequence[T_max] = p

    # Backward analytic costate recursion
    for k in range(T_max - 1, -1, -1):
        P = (
            2.0 * H
            + A_cl.T @ P @ A_cl
        )

        p = (
            -2.0 * Q @ x_ref
            + A_cl.T @ p
        )

        P_sequence[k] = P
        p_sequence[k] = p

    expected_gradient = jnp.zeros_like(K)

    # Forward moment recursion
    for k in range(T_max):
        P_next = P_sequence[k + 1]
        p_next = p_sequence[k + 1]

        # lambda_{k+1}
        # = P_{k+1} A_cl x_k + p_{k+1}
        #
        # Therefore:
        #
        # E[lambda_{k+1} x_k^T]
        # = P_{k+1} A_cl E[x_k x_k^T]
        #   + p_{k+1} E[x_k]^T.
        expected_lambda_x = (
            P_next @ A_cl @ S_k
            + jnp.outer(p_next, mu_k)
        )

        expected_gradient = (
            expected_gradient
            + 2.0 * R @ K @ S_k
            - B.T @ expected_lambda_x
        )

        mu_k = A_cl @ mu_k
        S_k = A_cl @ S_k @ A_cl.T

    return expected_gradient


gradient_analytic = exact_expected_gradient_uniform(
    K,
    x0_low,
    x0_high,
)


# ============================================================
# 9. Independent analytic-gradient check using autodiff
# ============================================================

def exact_expected_cost_uniform(K, low, high):
    """
    Exact E[J_T] using the first two moments of x_0.
    """
    A_cl = closed_loop_matrix(K)

    mu_k, S_k = uniform_initial_moments(
        low,
        high,
    )

    H = Q + K.T @ R @ K

    reference_constant = (
        x_ref.T @ Q @ x_ref
    )

    expected_cost = 0.0

    for _ in range(T_max):
        expected_stage_cost = (
            jnp.trace(H @ S_k)
            - 2.0 * x_ref.T @ Q @ mu_k
            + reference_constant
        )

        expected_cost = (
            expected_cost
            + expected_stage_cost
        )

        mu_k = A_cl @ mu_k
        S_k = A_cl @ S_k @ A_cl.T

    return expected_cost


gradient_analytic_autodiff = jax.grad(
    exact_expected_cost_uniform,
    argnums=0,
)(
    K,
    x0_low,
    x0_high,
)


# ============================================================
# 10. Numerical comparison
# ============================================================

gradient_error = (
    gradient_deer_mc
    - gradient_analytic
)

absolute_error = jnp.linalg.norm(
    gradient_error,
    ord="fro",
)

relative_error = (
    absolute_error
    / jnp.maximum(
        jnp.linalg.norm(
            gradient_analytic,
            ord="fro",
        ),
        1e-14,
    )
)

analytic_verification_error = jnp.linalg.norm(
    gradient_analytic
    - gradient_analytic_autodiff,
    ord="fro",
)

A_cl = closed_loop_matrix(K)

closed_loop_eigenvalues = jnp.linalg.eigvals(
    A_cl
)

spectral_radius = jnp.max(
    jnp.abs(closed_loop_eigenvalues)
)


print("Closed-loop eigenvalues:")
print(closed_loop_eigenvalues)

print("\nClosed-loop spectral radius:")
print(spectral_radius)

if spectral_radius >= 1.0:
    print(
        "\nWARNING: The sampled K is not asymptotically "
        "stabilizing. The finite-horizon computation is still "
        "defined, but trajectories and gradients may grow."
    )

print("\nDEER Monte Carlo expected gradient:")
print(gradient_deer_mc)

print("\nExact analytic expected gradient:")
print(gradient_analytic)

print("\nAutodiff check of analytic gradient:")
print(gradient_analytic_autodiff)

print("\nMonte Carlo componentwise standard error:")
print(gradient_deer_standard_error)

print("\nAbsolute Frobenius error:")
print(absolute_error)

print("\nRelative Frobenius error:")
print(relative_error)

print("\nAnalytic recursion vs. autodiff error:")
print(analytic_verification_error)

# print("\nMean DEER Newton steps:")
# print(float(np.mean(np.asarray(deer_newton_steps))))
# print("\nMean DEER Newton steps:")
# print(float(np.mean(np.asarray(deer_newton_steps))))

print("\nMaximum DEER state-rollout relative error:")
print(jnp.max(deer_state_errors))


# ============================================================
# 11. Monte Carlo convergence versus number of samples
# ============================================================

cumulative_gradient = jnp.cumsum(
    deer_sample_gradients,
    axis=0,
)

sample_counts = np.unique(
    np.geomspace(
        1,
        num_samples,
        num=15,
    ).astype(int)
)

relative_errors_by_sample_count = []

for count in sample_counts:
    gradient_at_count = (
        cumulative_gradient[count - 1]
        / count
    )

    error_at_count = (
        jnp.linalg.norm(
            gradient_at_count
            - gradient_analytic,
            ord="fro",
        )
        / jnp.maximum(
            jnp.linalg.norm(
                gradient_analytic,
                ord="fro",
            ),
            1e-14,
        )
    )

    relative_errors_by_sample_count.append(
        float(error_at_count)
    )


# ============================================================
# 12. Visualization 1:
#     Monte Carlo convergence
# ============================================================

plt.figure(figsize=(7, 5))

plt.loglog(
    sample_counts,
    relative_errors_by_sample_count,
    marker="o",
)

# Reference O(N^{-1/2}) rate
reference_rate = (
    relative_errors_by_sample_count[0]
    / np.sqrt(sample_counts)
)

plt.plot(
    sample_counts,
    reference_rate,
    linestyle="--",
    label=r"$Reference",
)

plt.xlabel("Number of sampled initial states")
plt.ylabel("Relative gradient error")
plt.title("DEER Monte Carlo gradient convergence")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 13. Visualization 2:
#     DEER Monte Carlo gradient matrix
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    np.asarray(gradient_deer_mc),
    aspect="auto",
)

plt.colorbar(label="Gradient value")
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
plt.title("DEER Monte Carlo expected gradient")
plt.tight_layout()
plt.show()


# ============================================================
# 14. Visualization 3:
#     Analytic gradient matrix
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    np.asarray(gradient_analytic),
    aspect="auto",
)

plt.colorbar(label="Gradient value")
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
plt.title("Exact analytic expected gradient")
plt.tight_layout()
plt.show()


# ============================================================
# 15. Visualization 4:
#     Absolute componentwise error
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
# 16. Visualization 5:
#     Componentwise DEER vs. analytic comparison
# ============================================================

gradient_mc_flat = np.asarray(
    gradient_deer_mc
).reshape(-1)

gradient_analytic_flat = np.asarray(
    gradient_analytic
).reshape(-1)

comparison_min = min(
    gradient_mc_flat.min(),
    gradient_analytic_flat.min(),
)

comparison_max = max(
    gradient_mc_flat.max(),
    gradient_analytic_flat.max(),
)

plt.figure(figsize=(6, 5))

plt.scatter(
    gradient_analytic_flat,
    gradient_mc_flat,
)

plt.plot(
    [comparison_min, comparison_max],
    [comparison_min, comparison_max],
    linestyle="--",
)

plt.xlabel("Analytic gradient component")
plt.ylabel("DEER Monte Carlo gradient component")
plt.title("DEER versus analytic gradient")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# 17. Optional policy update
# ============================================================

learning_rate = 1e-4

K_new = (
    K
    - learning_rate * gradient_deer_mc
)

print("\nUpdated feedback matrix K:")
print(K_new)