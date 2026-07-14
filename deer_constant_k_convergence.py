"""
lqr_policy_optimization_one_pass_deer_lqr.py

Policy-gradient optimization for finite-horizon LQR using a decoupled
one-pass DEER trajectory evaluation with a fixed LQR Jacobian.

Workflow:
    1. Compute analytic DARE gain K_star.
    2. Optimize a constant feedback gain K using Monte Carlo policy gradients.
    3. Estimate gradients using decoupled one-pass stacked DEER:
            z_i = [x_i, lambda_{T-i}]
       with fixed Jacobian:
            J_z = block_diag(A - B K, (A - B K)^T).
    4. Compare learned K against analytic K_star.

Requires:
    deer_fixed_j.py

If you saved the fixed-J file under a different name, change the import below.
"""

import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solve_discrete_are
from deer_LQR import deer_alg_fixed_j


# ============================================================
# 1. Configuration
# ============================================================

SEED = 0

T_HORIZON = 1000
NUM_MC_SAMPLES = 16
NUM_POLICY_ITERS = 60

DEER_MAX_ITERS = 20
DEER_TOL = 1e-9

INITIAL_STEP_SIZE = 3e-2
BACKTRACK_FACTOR = 0.5
MAX_BACKTRACK_STEPS = 18
MIN_STEP_SIZE = 1e-10

STABILITY_LIMIT = 0.999

RESAMPLE_EACH_ITERATION = False
USE_ANTITHETIC_SAMPLING = True

X0_LOW = -1.0
X0_HIGH = 1.0

PLOT_DIR = Path("lqr_one_pass_deer_lqr_results")
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
# 3. DARE solution and analytic optimal gain
# ============================================================

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
# 4. Basic dynamics, costs, and exact moments
# ============================================================

def closed_loop_matrix(K):
    return A - B @ K


def spectral_radius(K):
    eigvals = jnp.linalg.eigvals(closed_loop_matrix(K))
    return jnp.max(jnp.abs(eigvals))


def grad_stage_cost_x(x, K):
    """
    l(x,K) = x'Qx + u'Ru, u=-Kx.

    grad_x l = 2(Q + K' R K)x.
    """
    return 2.0 * (Q + K.T @ R @ K) @ x


def rollout_state(x0, K):
    """
    Return:
        x_traj = [x_0, ..., x_{T-1}], shape (T_HORIZON, n)
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
# 7. Robust DEER result parser
# ============================================================

def parse_deer_result(result):
    if not isinstance(result, tuple):
        raise TypeError(
            "deer_alg_fixed_j was expected to return a tuple, but returned "
            f"{type(result)}."
        )

    if len(result) < 2:
        raise ValueError(
            "deer_alg_fixed_j returned fewer than two values; the trajectory "
            "cannot be extracted."
        )

    z_deer = result[1]
    newton_steps = result[2] if len(result) > 2 else None

    return z_deer, newton_steps


# ============================================================
# 8. Decoupled one-pass fixed-J DEER gradient for one initial state
# ============================================================

def deer_lqr_one_pass_gradient_single(x0, K, guess_key):
    """
    Compute grad_K J_T(x0,K) using decoupled one-pass fixed-J DEER.

    Stacked variable:
        z_i = [x_i, lambda_{T-i}].

        lambda_T = 0

    The one-pass map is:
        x_{i+1} = A_cl x_i,
        lambda_{T-i-1} = grad_x l(x_{T-i-1},K)
                         + A_cl' lambda_{T-i}.

    This is decoupled because the costate driver x_{T-i-1} is fixed
    from the rollout, and the term G_x Delta x is omitted.
    """
    A_cl = closed_loop_matrix(K)

    dummy_inputs = jnp.zeros((T_HORIZON, CONTROL_DIM))

    lambda_T = jnp.zeros(STATE_DIM)

    def stacked_f(z, driver_x):
        x_i = z[:STATE_DIM]
        lambda_T_minus_i = z[STATE_DIM:]

        x_next = A_cl @ x_i

        lambda_previous = (
            grad_stage_cost_x(driver_x, K)
            + A_cl.T @ lambda_T_minus_i
        )

        return jnp.concatenate([x_next, lambda_previous])

    # Fixed Jacobian of stacked_f wrt z.
    # Because driver_x is fixed, this is block diagonal:
    #     d x_next / d x_i = A_cl
    #     d lambda_previous / d lambda_T_minus_i = A_cl.T
    #     d lambda_previous / d x_i = 0 in the decoupled approximation.
    J_stacked = jnp.block([
        [A_cl, jnp.zeros((STATE_DIM, STATE_DIM))],
        [jnp.zeros((STATE_DIM, STATE_DIM)), A_cl.T],
    ])

    z0 = jnp.concatenate([x0, lambda_T])

    # Small random guess to match DEER usage. Since the map is linear
    # and J is exact, convergence should be very fast.
    z_guess = 1e-2 * jr.normal(
        guess_key,
        shape=(T_HORIZON, 2 * STATE_DIM),
    )

    result = deer_alg_fixed_j(
        stacked_f,
        J_stacked,
        z0,
        z_guess,
        dummy_inputs,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )

    z_deer, newton_steps = parse_deer_result(result)

    x_deer = z_deer[:, :STATE_DIM]
    lambda_reversed = z_deer[:, STATE_DIM:]

    # Convert [lambda_{T-1}, ..., lambda_0] to [lambda_0, ..., lambda_{T-1}].
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)

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
        x_deer,
        lambda_k_plus_1,
    )

    gradient = jnp.sum(gradient_terms, axis=0)

    return gradient, newton_steps


# ============================================================
# 9. Sampling
# ============================================================

def sample_initial_states(key, sample_count):
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
# 10. Monte Carlo DEER expected gradient
# ============================================================

def deer_lqr_monte_carlo_gradient(K, x0_samples, key):
    print('computing grad from Monte Carlo...')
    sample_count = int(x0_samples.shape[0])
    guess_keys = jr.split(key, sample_count)

    gradients = []
    valid_newton_steps = []

    for index in range(sample_count):
        gradient_i, steps_i = deer_lqr_one_pass_gradient_single(
            x0_samples[index],
            K,
            guess_keys[index],
        )
        # print(f'grad {index} computed')
        gradients.append(gradient_i)

        if steps_i is not None:
            steps_array = np.asarray(steps_i)
            if steps_array.ndim == 0:
                valid_newton_steps.append(float(steps_array))

    gradients = jnp.stack(gradients, axis=0)

    mean_gradient = jnp.mean(gradients, axis=0)

    standard_error = jnp.zeros_like(mean_gradient)

    mean_steps = (
        float(np.mean(valid_newton_steps))
        if valid_newton_steps
        else None
    )
    print('Done computing Monte Carlo DEER gradient.')
    return (
        mean_gradient,
        standard_error,
        mean_steps,
    )


# ============================================================
# 11. Initial stabilizing gain
# ============================================================

K_initial = 0.5 * K_star

if float(spectral_radius(K_initial)) >= STABILITY_LIMIT:
    raise RuntimeError("The chosen initial gain is not strictly stabilizing.")


# ============================================================
# 12. Policy-gradient optimization
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
    "K_error": [],
    "K_relative_error": [],
    "gradient_norm_deer_lqr": [],
    "spectral_radius": [],
    "step_size": [],
    "mean_newton_steps": [],
    "gradient_standard_error_norm": [],
}

print("Analytic DARE optimal gain K_star:\n", np.asarray(K_star))
print("\nInitial gain K_0:\n", np.asarray(K_initial))
print("\nInitial ||K_0-K_star||_F:", float(jnp.linalg.norm(K_initial - K_star, ord="fro")))
print("Initial spectral radius:", float(spectral_radius(K_initial)))

start_time = time.perf_counter()

for iteration in range(NUM_POLICY_ITERS):
    print(f"\n=== Policy-gradient iteration {iteration} ===")
    iteration_key = jr.fold_in(optimization_key, iteration)

    if RESAMPLE_EACH_ITERATION:
        sample_key = jr.fold_in(sampling_key, iteration)
        x0_samples = sample_initial_states(sample_key, NUM_MC_SAMPLES)
    else:
        x0_samples = fixed_samples

    deer_gradient, deer_se, mean_steps = (
        deer_lqr_monte_carlo_gradient(
            K,
            x0_samples,
            iteration_key,
        )
    )
    
    step_size = INITIAL_STEP_SIZE

    K = K + step_size*deer_gradient

    current_radius = float(spectral_radius(K))

    K_error = float(jnp.linalg.norm(K - K_star, ord="fro"))
    K_relative_error = float(
        K_error / jnp.maximum(jnp.linalg.norm(K_star, ord="fro"), 1e-14)
    )

    deer_gradient_norm = float(jnp.linalg.norm(deer_gradient, ord="fro"))
    deer_se_norm = float(jnp.linalg.norm(deer_se, ord="fro"))

    history["iteration"].append(iteration)
    history["K_error"].append(K_error)
    history["K_relative_error"].append(K_relative_error)
    history["gradient_norm_deer_lqr"].append(deer_gradient_norm)
    history["spectral_radius"].append(current_radius)
    history["mean_newton_steps"].append(np.nan if mean_steps is None else mean_steps)
    history["gradient_standard_error_norm"].append(deer_se_norm)
    history["step_size"].append(step_size)

    if (
        iteration == 0
        or (iteration + 1) % 5 == 0
        or iteration + 1 == NUM_POLICY_ITERS
    ):
        print(
            f"Iteration {iteration:3d} | "
            f"||K-K*||_F={K_error:.6e} | "
            f"relK={K_relative_error:.3e} | "
            f"||g_DEER_LQR||_F={deer_gradient_norm:.6e} | "
            f"rho={current_radius:.6f} | "
            f"step={step_size:.3e}"
        )


    if K_error < 1e-8 and deer_gradient_norm < 1e-8:
        print("Stopping: gain and gradient tolerances were reached.")
        break

elapsed = time.perf_counter() - start_time


# ============================================================
# 13. Final comparison: analytic K_star vs learned K
# ============================================================

final_K_error = float(jnp.linalg.norm(K - K_star, ord="fro"))
final_K_relative_error = float(
    final_K_error / jnp.maximum(jnp.linalg.norm(K_star, ord="fro"), 1e-14)
)
final_radius = float(spectral_radius(K))

print("\n================ Final Gain Comparison ================\n")

print("Learned gain K:\n", np.asarray(K))
print("\nAnalytic DARE gain K_star:\n", np.asarray(K_star))
print("\nGain difference K - K_star:\n", np.asarray(K - K_star))
print("\nAbsolute gain error |K-K_star|:\n", np.abs(np.asarray(K - K_star)))

print("\nFinal ||K-K_star||_F:", final_K_error)
print("Final relative ||K-K_star||_F:", final_K_relative_error)
print("Final spectral radius:", final_radius)
print("Total optimization time:", elapsed, "seconds")


# ============================================================
# 14. Test-state rollouts for visualization
# ============================================================

def rollout_full_state(x0, K):
    x_traj, x_T = rollout_state(x0, K)
    return jnp.vstack([x_traj, x_T[None, :]])


x0_test = jnp.array([1.0, -0.75, 0.5])

trajectory_initial = rollout_full_state(x0_test, K_initial)
trajectory_final = rollout_full_state(x0_test, K)
trajectory_optimal = rollout_full_state(x0_test, K_star)

time_axis = np.arange(T_HORIZON + 1)
iterations = np.asarray(history["iteration"])


# ============================================================
# 15. Save numerical results
# ============================================================

np.savez(
    PLOT_DIR / "lqr_one_pass_deer_lqr_results.npz",
    K_learned=np.asarray(K),
    K_star=np.asarray(K_star),
    K_error=np.asarray(K - K_star),
    K_error_history=np.asarray(history["K_error"]),
    K_relative_error_history=np.asarray(history["K_relative_error"]),
    spectral_radius=np.asarray(history["spectral_radius"]),
    step_size=np.asarray(history["step_size"]),
    gradient_standard_error_norm=np.asarray(history["gradient_standard_error_norm"]),
)


# ============================================================
# 16. Plots
# ============================================================


plt.figure(figsize=(7, 5))
plt.semilogy(iterations, history["K_error"], marker="o", markersize=3)
plt.xlabel("Policy-gradient iteration")
plt.ylabel(r"$\|K-K^\star\|_F$")
plt.title("Convergence to analytic DARE gain")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "gain_error_convergence.png", dpi=200)
plt.show()

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
    label="Analytic K*",
)
plt.xlabel("Time step")
plt.ylabel(r"$\|x_k\|_2$")
plt.title("Closed-loop state trajectories")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "state_norm_comparison.png", dpi=200)
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(np.asarray(K), aspect="auto")
plt.colorbar(label="Gain value")
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Learned gain K")
plt.tight_layout()
plt.savefig(PLOT_DIR / "learned_gain.png", dpi=200)
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(np.asarray(K_star), aspect="auto")
plt.colorbar(label="Gain value")
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Analytic DARE gain K*")
plt.tight_layout()
plt.savefig(PLOT_DIR / "analytic_gain.png", dpi=200)
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(np.abs(np.asarray(K - K_star)), aspect="auto")
plt.colorbar(label="Absolute error")
plt.xlabel("State component")
plt.ylabel("Control component")
plt.title("Absolute learned-gain error")
plt.tight_layout()
plt.savefig(PLOT_DIR / "gain_absolute_error.png", dpi=200)
plt.show()

print(f"\nPlots and numerical results were saved to: {PLOT_DIR.resolve()}")