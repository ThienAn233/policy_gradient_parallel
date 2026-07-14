import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.linalg import solve_discrete_are
from deer import deer_alg


# ============================================================
# 1. Configuration
# ============================================================

SEED = 0

# Finite horizon used by DEER and by the analytic comparison.
T_HORIZON = 1000

# Number of initial states used in each DEER Monte Carlo gradient.
# Runtime scales approximately linearly with this number.
NUM_MC_SAMPLES = 64

# Number of policy-gradient iterations.
NUM_POLICY_ITERS = 60

# DEER settings.
DEER_MAX_ITERS = 20
DEER_TOL = 1e-9

# Backtracking gradient-descent settings.
INITIAL_STEP_SIZE = 3e-1
BACKTRACK_FACTOR = 0.5
MAX_BACKTRACK_STEPS = 18
MIN_STEP_SIZE = 1e-10

# Keep the learned closed loop strictly stable during optimization.
STABILITY_LIMIT = 0.999

# Use the same uniformly sampled initial states at every policy iteration.
# Common random numbers make the convergence curves less noisy.
RESAMPLE_EACH_ITERATION = False

# Antithetic sampling keeps the empirical mean exactly zero while every
# individual sample still has the requested uniform marginal distribution.
USE_ANTITHETIC_SAMPLING = True

# Uniform initial-state distribution.
X0_LOW = -1.0
X0_HIGH = 1.0

# Output directory for figures.
PLOT_DIR = Path("deer_constant_k_results")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Linear system and quadratic cost
# ============================================================

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

STATE_DIM = A.shape[0]
CONTROL_DIM = B.shape[1]

Q = jnp.eye(STATE_DIM)
R = jnp.eye(CONTROL_DIM)


# ============================================================
# 3. DARE solution and exact optimal constant gain
# ============================================================

# The DARE terminal matrix is used as the finite-horizon terminal cost:
#
#   J_T(K) = E[ sum_{k=0}^{T-1} (x_k'Qx_k + u_k'Ru_k)
#               + x_T' P_star x_T ],
#   u_k = -K x_k.
#
# With this terminal cost, K_star is the exact minimizer even for a finite
# horizon, because P_star is the Bellman fixed point.
P_star_np = solve_discrete_are(
    np.asarray(A),
    np.asarray(B),
    np.asarray(Q),
    np.asarray(R),
)

K_star_np = np.linalg.solve(
    np.asarray(R) + np.asarray(B).T @ P_star_np @ np.asarray(B),
    np.asarray(B).T @ P_star_np @ np.asarray(A),
)

P_star = jnp.asarray(P_star_np)
K_star = jnp.asarray(K_star_np)


# ============================================================
# 4. Basic dynamics, costs, and moments
# ============================================================


def closed_loop_matrix(K):
    """Closed-loop matrix for u=-Kx."""
    return A - B @ K



def spectral_radius(K):
    """Spectral radius of A-BK."""
    eigvals = jnp.linalg.eigvals(closed_loop_matrix(K))
    return jnp.max(jnp.abs(eigvals))



def stage_cost(x, K):
    """l(x,K)=x'Qx+u'Ru with u=-Kx."""
    u = -K @ x
    return x.T @ Q @ x + u.T @ R @ u



def grad_stage_cost_x(x, K):
    """Partial derivative of l(x,K) with respect to x."""
    return 2.0 * (Q + K.T @ R @ K) @ x



def direct_stage_gradient_K(x, K):
    """Partial derivative of l(x,K) with respect to K."""
    return 2.0 * R @ K @ jnp.outer(x, x)



def rollout_state(x0, K):
    """
    Return
        x_traj = [x_0, ..., x_{T-1}], shape (T_HORIZON,n)
        x_T    = terminal state.
    """
    A_cl = closed_loop_matrix(K)

    def step(x_k, _):
        x_next = A_cl @ x_k
        return x_next, x_k

    x_T, x_traj = jax.lax.scan(
        step,
        x0,
        xs=None,
        length=T_HORIZON,
    )

    return x_traj, x_T



def uniform_second_moment(low, high, dimension):
    """E[x_0 x_0^T] for independent uniform components."""
    low_vec = jnp.full((dimension,), low)
    high_vec = jnp.full((dimension,), high)

    mean = 0.5 * (low_vec + high_vec)
    variance = ((high_vec - low_vec) ** 2) / 12.0

    return jnp.diag(variance) + jnp.outer(mean, mean)


S0_EXACT = uniform_second_moment(
    X0_LOW,
    X0_HIGH,
    STATE_DIM,
)


# ============================================================
# 5. Exact expected finite-horizon cost
# ============================================================


def exact_expected_cost(K):
    """
    Exact expected cost under the uniform initial-state distribution:

        E[J_T(K)]
        = E[sum_{k=0}^{T-1} x_k'(Q+K'RK)x_k
            + x_T'P_star x_T].
    """
    A_cl = closed_loop_matrix(K)
    H = Q + K.T @ R @ K

    S_k = S0_EXACT
    expected_cost = 0.0

    for _ in range(T_HORIZON):
        expected_cost = expected_cost + jnp.trace(H @ S_k)
        S_k = A_cl @ S_k @ A_cl.T

    expected_cost = expected_cost + jnp.trace(P_star @ S_k)
    return expected_cost


exact_expected_cost_jit = jax.jit(exact_expected_cost)
exact_expected_gradient_autodiff = jax.jit(jax.grad(exact_expected_cost))


# ============================================================
# 6. Analytic expected policy gradient
# ============================================================


def analytic_expected_gradient(K):
    """
    Exact expected policy gradient for the same finite-horizon problem.

    If lambda_k=2 P_k x_k, then

        P_T = P_star,
        P_k = Q + K'RK + A_cl' P_{k+1} A_cl.

    The gradient is

        grad_K J = 2 sum_k
            (R K - B' P_{k+1} A_cl) E[x_k x_k'].
    """
    A_cl = closed_loop_matrix(K)
    H = Q + K.T @ R @ K

    P_sequence = [None] * (T_HORIZON + 1)
    P_sequence[T_HORIZON] = P_star

    P_next = P_star
    for k in range(T_HORIZON - 1, -1, -1):
        P_k = H + A_cl.T @ P_next @ A_cl
        P_sequence[k] = P_k
        P_next = P_k

    S_k = S0_EXACT
    gradient = jnp.zeros_like(K)

    for k in range(T_HORIZON):
        P_next = P_sequence[k + 1]

        gradient = gradient + 2.0 * (
            R @ K - B.T @ P_next @ A_cl
        ) @ S_k

        S_k = A_cl @ S_k @ A_cl.T

    return gradient


analytic_expected_gradient_jit = jax.jit(analytic_expected_gradient)


# ============================================================
# 7. Robust DEER result parser
# ============================================================


def parse_deer_result(result):
    """
    Extract the trajectory and, when available, the Newton iteration count.
    Some deer versions return None for the count.
    """
    if not isinstance(result, tuple):
        raise TypeError(
            "deer_alg was expected to return a tuple, but returned "
            f"{type(result)}."
        )

    if len(result) < 2:
        raise ValueError(
            "deer_alg returned fewer than two values; the trajectory "
            "cannot be extracted."
        )

    z_deer = result[1]
    newton_steps = result[2] if len(result) > 2 else None

    return z_deer, newton_steps


# ============================================================
# 8. DEER gradient for one initial state
# ============================================================


def deer_gradient_single(x0, K, guess_key):
    """
    Compute grad_K J_T(x0,K) using the user's stacked DEER construction:

        z_i = [x_i, lambda_{T-i}].

    The terminal cost is x_T'P_star x_T, hence

        lambda_T = 2 P_star x_T.
    """
    A_cl = closed_loop_matrix(K)

    # Chronological state trajectory x_0,...,x_{T-1} and terminal x_T.
    x_traj, x_T = rollout_state(x0, K)

    # Costate driver x_{T-1},...,x_0.
    reversed_x_traj = jnp.flip(x_traj, axis=0)

    def stacked_f(z, driver_x):
        x_i = z[:STATE_DIM]
        lambda_T_minus_i = z[STATE_DIM:]

        # Forward state dynamics.
        x_next = A_cl @ x_i

        # Backward costate dynamics.
        lambda_previous = (
            grad_stage_cost_x(driver_x, K)
            + A_cl.T @ lambda_T_minus_i
        )

        return jnp.concatenate([x_next, lambda_previous])

    lambda_T = 2.0 * P_star @ x_T

    # The stacked boundary combines x_0 and lambda_T.
    z0 = jnp.concatenate([x0, lambda_T])

    # The system is linear, so a zero or random initial trajectory guess works.
    # A small random guess is kept to match the original implementation.
    z_guess = 1e-2 * jr.normal(
        guess_key,
        shape=(T_HORIZON, 2 * STATE_DIM),
    )

    result = deer_alg(
        stacked_f,
        z0,
        z_guess,
        reversed_x_traj,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )

    z_deer, newton_steps = parse_deer_result(result)

    # State block: x_1,...,x_T.
    x_deer = z_deer[:, :STATE_DIM]

    # Costate block: lambda_{T-1},...,lambda_0.
    lambda_reversed = z_deer[:, STATE_DIM:]

    # Convert to lambda_0,...,lambda_{T-1}.
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)

    # Pair x_k with lambda_{k+1}, k=0,...,T-1.
    lambda_k_plus_1 = jnp.vstack([
        lambda_chronological[1:],
        lambda_T[None, :],
    ])

    def gradient_step(x_k, lambda_next):
        # For u=-Kx:
        # grad_K H_k = (2 R K x_k - B'lambda_{k+1}) x_k'.
        return jnp.outer(
            2.0 * R @ K @ x_k - B.T @ lambda_next,
            x_k,
        )

    gradient_terms = jax.vmap(gradient_step)(
        x_traj,
        lambda_k_plus_1,
    )

    gradient = jnp.sum(gradient_terms, axis=0)

    # Check that DEER's state block agrees with the direct rollout.
    rollout_states_1_to_T = jnp.vstack([
        x_traj[1:],
        x_T[None, :],
    ])

    state_relative_error = (
        jnp.linalg.norm(x_deer - rollout_states_1_to_T)
        / jnp.maximum(
            jnp.linalg.norm(rollout_states_1_to_T),
            1e-14,
        )
    )

    return gradient, newton_steps, state_relative_error


# ============================================================
# 9. Uniform and antithetic sampling
# ============================================================


def sample_initial_states(key, sample_count):
    """Sample x_0 from the requested independent uniform distribution."""
    if not USE_ANTITHETIC_SAMPLING:
        return jr.uniform(
            key,
            shape=(sample_count, STATE_DIM),
            minval=X0_LOW,
            maxval=X0_HIGH,
        )

    if not np.isclose(X0_LOW, -X0_HIGH):
        raise ValueError(
            "Antithetic sampling requires symmetric bounds. Set "
            "USE_ANTITHETIC_SAMPLING=False for nonsymmetric bounds."
        )

    half_count = sample_count // 2

    positive_half = jr.uniform(
        key,
        shape=(half_count, STATE_DIM),
        minval=X0_LOW,
        maxval=X0_HIGH,
    )

    samples = jnp.concatenate([
        positive_half,
        -positive_half,
    ], axis=0)

    if sample_count % 2 == 1:
        extra_key = jr.fold_in(key, 1)
        extra = jr.uniform(
            extra_key,
            shape=(1, STATE_DIM),
            minval=X0_LOW,
            maxval=X0_HIGH,
        )
        samples = jnp.concatenate([samples, extra], axis=0)

    return samples


# ============================================================
# 10. DEER Monte Carlo expected gradient
# ============================================================


def deer_monte_carlo_gradient(K, x0_samples, key):
    """Average the DEER gradients over sampled initial states."""
    sample_count = int(x0_samples.shape[0])
    guess_keys = jr.split(key, sample_count)

    gradients = []
    state_errors = []
    valid_newton_steps = []

    for index in range(sample_count):
        gradient_i, steps_i, state_error_i = deer_gradient_single(
            x0_samples[index],
            K,
            guess_keys[index],
        )

        gradients.append(gradient_i)
        state_errors.append(state_error_i)

        if steps_i is not None:
            steps_array = np.asarray(steps_i)
            if steps_array.ndim == 0:
                valid_newton_steps.append(float(steps_array))

    gradients = jnp.stack(gradients, axis=0)
    state_errors = jnp.asarray(state_errors)

    mean_gradient = jnp.mean(gradients, axis=0)

    if sample_count > 1:
        standard_error = (
            jnp.std(gradients, axis=0, ddof=1)
            / jnp.sqrt(sample_count)
        )
    else:
        standard_error = jnp.zeros_like(mean_gradient)

    mean_steps = (
        float(np.mean(valid_newton_steps))
        if valid_newton_steps
        else None
    )

    return (
        mean_gradient,
        standard_error,
        mean_steps,
        float(jnp.max(state_errors)),
    )


# ============================================================
# 11. Initial stabilizing gain
# ============================================================

# Starting at half of K_star gives a stabilizing gain that is visibly
# different from the optimum.
K_initial = 0.5 * K_star

if float(spectral_radius(K_initial)) >= STABILITY_LIMIT:
    raise RuntimeError(
        "The chosen initial gain is not strictly stabilizing."
    )


# ============================================================
# 12. Policy-gradient optimization using DEER gradients
# ============================================================

master_key = jr.PRNGKey(SEED)
sampling_key, optimization_key = jr.split(master_key)

fixed_samples = sample_initial_states(
    sampling_key,
    NUM_MC_SAMPLES,
)

K = K_initial

history = {
    "iteration": [],
    "cost": [],
    "K_error": [],
    "gradient_norm_deer": [],
    "deer_vs_analytic": [],
    "analytic_vs_autodiff": [],
    "spectral_radius": [],
    "step_size": [],
    "mean_newton_steps": [],
    "max_state_error": [],
}

print("DARE optimal constant gain K_star:\n", np.asarray(K_star))
print("\nInitial gain K_0:\n", np.asarray(K_initial))
print("\nInitial spectral radius:", float(spectral_radius(K_initial)))
print("Initial exact expected cost:", float(exact_expected_cost_jit(K_initial)))

for iteration in range(NUM_POLICY_ITERS):
    iteration_key = jr.fold_in(optimization_key, iteration)

    if RESAMPLE_EACH_ITERATION:
        sample_key = jr.fold_in(sampling_key, iteration)
        x0_samples = sample_initial_states(
            sample_key,
            NUM_MC_SAMPLES,
        )
    else:
        x0_samples = fixed_samples

    deer_gradient, deer_se, mean_steps, max_state_error = (
        deer_monte_carlo_gradient(
            K,
            x0_samples,
            iteration_key,
        )
    )

    analytic_gradient = analytic_expected_gradient_jit(K)
    autodiff_gradient = exact_expected_gradient_autodiff(K)

    current_cost = float(exact_expected_cost_jit(K))
    current_radius = float(spectral_radius(K))

    deer_vs_analytic = float(
        jnp.linalg.norm(deer_gradient - analytic_gradient, ord="fro")
        / jnp.maximum(
            jnp.linalg.norm(analytic_gradient, ord="fro"),
            1e-14,
        )
    )

    analytic_vs_autodiff = float(
        jnp.linalg.norm(analytic_gradient - autodiff_gradient, ord="fro")
    )

    K_error = float(jnp.linalg.norm(K - K_star, ord="fro"))
    deer_gradient_norm = float(jnp.linalg.norm(deer_gradient, ord="fro"))

    history["iteration"].append(iteration)
    history["cost"].append(current_cost)
    history["K_error"].append(K_error)
    history["gradient_norm_deer"].append(deer_gradient_norm)
    history["deer_vs_analytic"].append(deer_vs_analytic)
    history["analytic_vs_autodiff"].append(analytic_vs_autodiff)
    history["spectral_radius"].append(current_radius)
    history["mean_newton_steps"].append(
        np.nan if mean_steps is None else mean_steps
    )
    history["max_state_error"].append(max_state_error)

    # Backtracking uses the DEER Monte Carlo gradient as the direction.
    step_size = INITIAL_STEP_SIZE
    accepted = False

    for _ in range(MAX_BACKTRACK_STEPS):
        K_candidate = K - step_size * deer_gradient
        candidate_radius = float(spectral_radius(K_candidate))

        if candidate_radius < STABILITY_LIMIT:
            candidate_cost = float(exact_expected_cost_jit(K_candidate))

            if candidate_cost < current_cost:
                accepted = True
                break

        step_size *= BACKTRACK_FACTOR

        if step_size < MIN_STEP_SIZE:
            break

    history["step_size"].append(step_size if accepted else 0.0)

    if (
        iteration == 0
        or (iteration + 1) % 5 == 0
        or iteration + 1 == NUM_POLICY_ITERS
    ):
        print(
            f"Iteration {iteration:3d} | "
            f"cost={current_cost:.10f} | "
            f"||K-K*||_F={K_error:.6e} | "
            f"||g_DEER||_F={deer_gradient_norm:.6e} | "
            f"DEER/analytic={deer_vs_analytic:.3e} | "
            f"rho={current_radius:.6f} | "
            f"step={step_size if accepted else 0.0:.3e}"
        )

    if not accepted:
        print(
            "Stopping: the DEER Monte Carlo direction did not produce "
            "a stable cost-decreasing step. Increase NUM_MC_SAMPLES or "
            "reduce INITIAL_STEP_SIZE if this happens too early."
        )
        break

    K = K_candidate

    if K_error < 1e-8 and deer_gradient_norm < 1e-8:
        print("Stopping: gain and gradient tolerances were reached.")
        break


# Record the final point if it differs from the last recorded iterate.
final_cost = float(exact_expected_cost_jit(K))
final_K_error = float(jnp.linalg.norm(K - K_star, ord="fro"))
final_radius = float(spectral_radius(K))
final_analytic_gradient = analytic_expected_gradient_jit(K)
final_autodiff_gradient = exact_expected_gradient_autodiff(K)

print("\nFinal learned gain K:\n", np.asarray(K))
print("\nDARE optimal gain K_star:\n", np.asarray(K_star))
print("\nFinal ||K-K_star||_F:", final_K_error)
print("Final expected cost:", final_cost)
print("Optimal expected cost:", float(exact_expected_cost_jit(K_star)))
print("Final spectral radius:", final_radius)
print(
    "Final analytic-vs-autodiff gradient error:",
    float(jnp.linalg.norm(final_analytic_gradient - final_autodiff_gradient)),
)


# ============================================================
# 13. Test-state rollouts for visualization
# ============================================================


def rollout_full_state(x0, K):
    """Return x_0,...,x_T for plotting."""
    x_traj, x_T = rollout_state(x0, K)
    return jnp.vstack([x_traj, x_T[None, :]])


x0_test = jnp.array([1.0, -0.75, 0.5])

trajectory_initial = rollout_full_state(x0_test, K_initial)
trajectory_final = rollout_full_state(x0_test, K)
trajectory_optimal = rollout_full_state(x0_test, K_star)

time_axis = np.arange(T_HORIZON + 1)


# ============================================================
# 14. Plots
# ============================================================

iterations = np.asarray(history["iteration"])

# Expected cost convergence.
plt.figure(figsize=(7, 5))
plt.plot(iterations, history["cost"], marker="o", markersize=3)
plt.axhline(
    float(exact_expected_cost_jit(K_star)),
    linestyle="--",
    label="Optimal cost",
)
plt.xlabel("Policy-gradient iteration")
plt.ylabel("Exact expected finite-horizon cost")
plt.title("Cost convergence using DEER Monte Carlo gradients")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "cost_convergence.png", dpi=200)
plt.show()

# Distance to the optimal constant gain.
plt.figure(figsize=(7, 5))
plt.semilogy(iterations, history["K_error"], marker="o", markersize=3)
plt.xlabel("Policy-gradient iteration")
plt.ylabel(r"$\|K-K^\star\|_F$")
plt.title("Convergence to the DARE optimal gain")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "gain_error_convergence.png", dpi=200)
plt.show()

# Gradient validation.
plt.figure(figsize=(7, 5))
plt.semilogy(
    iterations,
    np.maximum(history["deer_vs_analytic"], 1e-18),
    label="DEER MC vs analytic",
)
plt.semilogy(
    iterations,
    np.maximum(history["analytic_vs_autodiff"], 1e-18),
    label="Analytic vs JAX autodiff",
)
plt.xlabel("Policy-gradient iteration")
plt.ylabel("Gradient discrepancy")
plt.title("Policy-gradient validation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "gradient_validation.png", dpi=200)
plt.show()

# Closed-loop spectral radius.
plt.figure(figsize=(7, 5))
plt.plot(iterations, history["spectral_radius"], marker="o", markersize=3)
plt.axhline(1.0, linestyle="--", label="Stability boundary")
plt.xlabel("Policy-gradient iteration")
plt.ylabel(r"$\rho(A-BK)$")
plt.title("Closed-loop stability during optimization")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "spectral_radius.png", dpi=200)
plt.show()

# State-norm comparison.
plt.figure(figsize=(7, 5))
plt.semilogy(
    time_axis,
    np.linalg.norm(np.asarray(trajectory_initial), axis=1),
    label="Initial K",
)
plt.semilogy(
    time_axis,
    np.linalg.norm(np.asarray(trajectory_final), axis=1),
    label="Learned K",
)
plt.semilogy(
    time_axis,
    np.linalg.norm(np.asarray(trajectory_optimal), axis=1),
    label="Optimal K*",
)
plt.xlabel("Time step")
plt.ylabel(r"$\|x_k\|_2$")
plt.title("Closed-loop state trajectories")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "state_norm_comparison.png", dpi=200)
plt.show()

# Learned gain heatmap.
plt.figure(figsize=(6, 5))
plt.imshow(np.asarray(K), aspect="auto")
plt.colorbar(label="Gain value")
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Learned constant gain K")
plt.tight_layout()
plt.savefig(PLOT_DIR / "learned_gain.png", dpi=200)
plt.show()

# Optimal gain heatmap.
plt.figure(figsize=(6, 5))
plt.imshow(np.asarray(K_star), aspect="auto")
plt.colorbar(label="Gain value")
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("DARE optimal gain K*")
plt.tight_layout()
plt.savefig(PLOT_DIR / "optimal_gain.png", dpi=200)
plt.show()

# Absolute gain error heatmap.
plt.figure(figsize=(6, 5))
plt.imshow(np.abs(np.asarray(K - K_star)), aspect="auto")
plt.colorbar(label="Absolute error")
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Absolute learned-gain error")
plt.tight_layout()
plt.savefig(PLOT_DIR / "gain_absolute_error.png", dpi=200)
plt.show()

print(f"\nPlots were saved to: {PLOT_DIR.resolve()}")
