"""
lqr_policy_optimization_two_pass_deer_lqr.py

Simple two-pass decoupled fixed-J DEER version.

Workflow:
    1. Compute analytic DARE gain K_star.
    2. Use forward DEER to compute x_1,...,x_T.
    3. Use backward DEER to compute lambda_{T-1},...,lambda_0.
    4. Compute the policy gradient and update K by gradient descent.
    5. Compare learned K with analytic K_star.

Requires:
    deer_LQR.py with deer_alg_fixed_j.
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

# Start small. Increase later if needed.
T_HORIZON = 20
NUM_MC_SAMPLES = 10
NUM_POLICY_ITERS = 60

# Fixed-J LQR is affine, so one DEER update is usually enough.
DEER_MAX_ITERS = 10
DEER_TOL = 1e-9

INITIAL_STEP_SIZE = 3e-2
STABILITY_LIMIT = 0.999

RESAMPLE_EACH_ITERATION = False
USE_ANTITHETIC_SAMPLING = True

K_TOL = 1e-4
GRAD_NORM_TOL = 1e-8

X0_LOW = -1.0
X0_HIGH = 1.0

PLOT_DIR = Path("lqr_two_pass_deer_lqr_results")
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
# 3. Analytic DARE gain K_star
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
# 4. Helper functions
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
    Sequential rollout for plotting only.

    Return:
        x_traj = [x_0, ..., x_{T-1}]
        x_T
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


def parse_deer_result(result):
    z_deer = result[1]
    newton_steps = result[2] if len(result) > 2 else None
    return z_deer, newton_steps


# ============================================================
# 5. Sampling
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
            "Antithetic sampling requires symmetric bounds."
        )

    half_count = sample_count // 2

    positive_half = jr.uniform(
        key,
        shape=(half_count, STATE_DIM),
        minval=X0_LOW,
        maxval=X0_HIGH,
    )

    samples = jnp.concatenate([positive_half, -positive_half], axis=0)

    if sample_count % 2 == 1:
        extra = jr.uniform(
            jr.fold_in(key, 1),
            shape=(1, STATE_DIM),
            minval=X0_LOW,
            maxval=X0_HIGH,
        )
        samples = jnp.concatenate([samples, extra], axis=0)

    return samples


# ============================================================
# 6. Two-pass decoupled DEER gradient for one initial state
# ============================================================

def deer_lqr_two_pass_gradient_single(x0, K, guess_key):
    """
    Two-pass decoupled fixed-J DEER.

    Pass 1:
        x_{k+1} = A_cl x_k

    Pass 2:
        lambda_k = grad_x l(x_k,K) + A_cl' lambda_{k+1}

    Terminal costate:
        lambda_T = 0.
    """
    A_cl = closed_loop_matrix(K)
    lambda_T = jnp.zeros(STATE_DIM)

    key_x, key_lam = jr.split(guess_key)

    # ------------------------------------------------------------
    # Forward DEER pass
    # ------------------------------------------------------------

    def forward_f(x, dummy):
        return A_cl @ x

    states_guess = jr.normal(
        key_x,
        shape=(T_HORIZON, STATE_DIM),
    )

    dummy_inputs = jnp.zeros((T_HORIZON, CONTROL_DIM))

    forward_result = deer_alg_fixed_j(
        forward_f,
        A_cl,
        x0,
        states_guess,
        dummy_inputs,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )

    states_deer, fwd_steps = parse_deer_result(forward_result)

    # states_deer = [x_1, ..., x_T]
    # x_traj      = [x_0, ..., x_{T-1}]
    x_traj = jnp.vstack([x0, states_deer[:-1]])

    # ------------------------------------------------------------
    # Backward DEER pass
    # ------------------------------------------------------------

    def backward_f(lambda_next, x_k):
        return grad_stage_cost_x(x_k, K) + A_cl.T @ lambda_next

    costate_guess =  jr.normal(
        key_lam,
        shape=(T_HORIZON, STATE_DIM),
    )

    x_traj_rev = jnp.flip(x_traj, axis=0)

    backward_result = deer_alg_fixed_j(
        backward_f,
        A_cl.T,
        lambda_T,
        costate_guess,
        x_traj_rev,
        num_iters=DEER_MAX_ITERS,
        full_trace=False,
        Ts=None,
        tol=DEER_TOL,
    )

    lambda_reversed, bwd_steps = parse_deer_result(backward_result)

    # lambda_reversed      = [lambda_{T-1}, ..., lambda_0]
    # lambda_chronological = [lambda_0, ..., lambda_{T-1}]
    lambda_chronological = jnp.flip(lambda_reversed, axis=0)

    lambda_k_plus_1 = jnp.vstack([
        lambda_chronological[1:],
        lambda_T[None, :],
    ])

    # ------------------------------------------------------------
    # Policy gradient
    # ------------------------------------------------------------

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

    return gradient, fwd_steps, bwd_steps


# ============================================================
# 7. Monte Carlo DEER gradient
# ============================================================

batched_deer_lqr_two_pass_gradient = jax.jit(
    jax.vmap(
        deer_lqr_two_pass_gradient_single,
        in_axes=(0, None, 0),
        out_axes=(0, 0, 0),
    )
)

def deer_lqr_monte_carlo_gradient(K, x0_samples, key):
    sample_count = int(x0_samples.shape[0])
    guess_keys = jr.split(key, sample_count)

    gradients = []
    fwd_steps_list = []
    bwd_steps_list = []

    gradients, _ , _ = batched_deer_lqr_two_pass_gradient(
        x0_samples,
        K,
        guess_keys,
    )

    gradients = jnp.stack(gradients, axis=0)
    mean_gradient = jnp.mean(gradients, axis=0)

    if sample_count > 1:
        standard_error = jnp.std(gradients, axis=0, ddof=1) / jnp.sqrt(sample_count)
    else:
        standard_error = jnp.zeros_like(mean_gradient)

    mean_fwd_steps = float(np.mean(fwd_steps_list)) if fwd_steps_list else None
    mean_bwd_steps = float(np.mean(bwd_steps_list)) if bwd_steps_list else None

    return mean_gradient, standard_error, mean_fwd_steps, mean_bwd_steps


# ============================================================
# 8. Policy-gradient optimization
# ============================================================

K_initial = 0.5 * K_star

if float(spectral_radius(K_initial)) >= STABILITY_LIMIT:
    raise RuntimeError("The chosen initial gain is not strictly stabilizing.")

master_key = jr.PRNGKey(SEED)
sampling_key, optimization_key = jr.split(master_key)

fixed_samples = sample_initial_states(sampling_key, NUM_MC_SAMPLES)

K = K_initial

history = {
    "iteration": [],
    "K_error": [],
    "K_relative_error": [],
    "gradient_norm": [],
    "spectral_radius": [],
    "step_size": [],
    "mean_fwd_steps": [],
    "mean_bwd_steps": [],
    "gradient_standard_error_norm": [],
}

print("Analytic DARE optimal gain K_star:\n", np.asarray(K_star))
print("\nInitial gain K_0:\n", np.asarray(K_initial))
print("\nInitial ||K_0-K_star||_F:", float(jnp.linalg.norm(K_initial - K_star, ord="fro")))
print("Initial spectral radius:", float(spectral_radius(K_initial)))

start_time = time.perf_counter()

for iteration in range(NUM_POLICY_ITERS):
    iteration_key = jr.fold_in(optimization_key, iteration)

    if RESAMPLE_EACH_ITERATION:
        x0_samples = sample_initial_states(
            jr.fold_in(sampling_key, iteration),
            NUM_MC_SAMPLES,
        )
    else:
        x0_samples = fixed_samples

    gradient, gradient_se, mean_fwd_steps, mean_bwd_steps = deer_lqr_monte_carlo_gradient(
        K,
        x0_samples,
        iteration_key,
    )

    step_size = INITIAL_STEP_SIZE

    # Gradient descent. Important: use minus, not plus.
    K = K - step_size * gradient

    current_radius = float(spectral_radius(K))

    K_error = float(jnp.linalg.norm(K - K_star, ord="fro"))
    K_relative_error = float(K_error / jnp.maximum(jnp.linalg.norm(K_star, ord="fro"), 1e-14))
    gradient_norm = float(jnp.linalg.norm(gradient, ord="fro"))
    gradient_se_norm = float(jnp.linalg.norm(gradient_se, ord="fro"))

    history["iteration"].append(iteration)
    history["K_error"].append(K_error)
    history["K_relative_error"].append(K_relative_error)
    history["gradient_norm"].append(gradient_norm)
    history["spectral_radius"].append(current_radius)
    history["step_size"].append(step_size)
    history["mean_fwd_steps"].append(np.nan if mean_fwd_steps is None else mean_fwd_steps)
    history["mean_bwd_steps"].append(np.nan if mean_bwd_steps is None else mean_bwd_steps)
    history["gradient_standard_error_norm"].append(gradient_se_norm)

    if (
        iteration == 0
        or (iteration + 1) % 5 == 0
        or iteration + 1 == NUM_POLICY_ITERS
    ):
        print(
            f"Iteration {iteration:3d} | "
            f"||K-K*||_F={K_error:.6e} | "
            f"relK={K_relative_error:.3e} | "
            f"||grad||_F={gradient_norm:.6e} | "
            f"rho={current_radius:.6f} | "
            f"step={step_size:.3e}"
        )

    if current_radius >= STABILITY_LIMIT:
        print("Stopping: closed-loop system became unstable.")
        break

    if K_error < K_TOL and gradient_norm < GRAD_NORM_TOL:
        print("Stopping: gain and gradient tolerances were reached.")
        break

elapsed = time.perf_counter() - start_time


# ============================================================
# 9. Final comparison
# ============================================================

final_K_error = float(jnp.linalg.norm(K - K_star, ord="fro"))
final_K_relative_error = float(final_K_error / jnp.maximum(jnp.linalg.norm(K_star, ord="fro"), 1e-14))
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
# 10. Plots
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

np.savez(
    PLOT_DIR / "lqr_two_pass_deer_lqr_results.npz",
    K_learned=np.asarray(K),
    K_star=np.asarray(K_star),
    K_error=np.asarray(K - K_star),
    K_error_history=np.asarray(history["K_error"]),
    K_relative_error_history=np.asarray(history["K_relative_error"]),
    spectral_radius=np.asarray(history["spectral_radius"]),
    step_size=np.asarray(history["step_size"]),
    gradient_standard_error_norm=np.asarray(history["gradient_standard_error_norm"]),
)

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