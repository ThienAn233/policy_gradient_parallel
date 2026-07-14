import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from deer import deer_alg

# --- System Definition ---
T_max = 300
A = jnp.array([[-1.0, 0.0, 0.0],
               [ 0.0,.7, 0.0],
               [ 0.0, 0.0,-0.2]])
B = jnp.array([[0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1],
               [0.1, 0.1, 0.1]])

# Initial Policy and State
key = jr.PRNGKey(0)
K = jax.random.normal(key, shape=(3, 3)) * 0.1 # Scaled down for stability
x0 = jnp.array([1.0, 1.0, 1.0])

# --- Dynamics ---
def f(x, u_dummy):
    # Discrete closed-loop dynamics: x_{k+1} = (A - BK)x_k
    return (A - B @ K) @ x

# --- Forward Pass (DEER + Manual Rollout) ---
states_guess = jax.random.normal(jr.PRNGKey(1), shape=(T_max, 3))
dummy_inputs = jnp.zeros((T_max, 3)) # No external inputs needed for closed-loop
tol = 1e-7

# DEER forward pass (using Ts=None for discrete map if your DEER version supports it)
_, states_deer, newton_steps, *_ = deer_alg(
    f, x0, states_guess, dummy_inputs,
    num_iters=T_max, full_trace=True, Ts=None, tol=tol
)

# Manual discrete rollout to verify
def rollout_step(x, _):
    x_next = f(x, None)
    return x_next, x_next

_, states_rollout = jax.lax.scan(rollout_step, x0, jnp.arange(T_max))
print("Forward Pass Error (DEER vs Rollout):", jnp.max(jnp.abs(states_deer - states_rollout)))

# --- Backward Pass: Costate Calculation ---
# We pad the state trajectory with x0 at the beginning to match indices
x_traj = jnp.vstack([x0, states_rollout[:-1]]) 

def backward_costate_step(lambda_next, x_k):
    """
    Computes lambda_k = grad_x l(x_k, K) + F_x^T lambda_{k+1}
    """
    u_k = -K @ x_k
    
    # \nabla_x l(x, K)
    grad_x_l = 2 * (x_k - 1.0) - 2 * K.T @ u_k
    
    # F_x^T = (A - BK)^T
    F_x_T = (A - B @ K).T
    
    # Costate recurrence
    lambda_k = grad_x_l + F_x_T @ lambda_next
    return lambda_k
def back_step(lambda_next, x_k):
    return backward_costate_step(lambda_next, x_k), backward_costate_step(lambda_next, x_k)
 
# The terminal costate lambda_T is usually 0 if there is no terminal cost, 
# or \nabla_x of the terminal cost. We will assume 0 here.
lambda_T = jnp.zeros(3)
costate_guess = jax.random.normal(jr.PRNGKey(2), shape=(T_max, 3))


# Run lax.scan backward by reversing the state trajectory
_, lambda_traj_rev = jax.lax.scan(back_step, lambda_T, jnp.flip(x_traj, axis=0))


_, costate_deer, newton_steps, *_ = deer_alg(
    backward_costate_step, lambda_T, costate_guess, jnp.flip(x_traj, axis=0),
    num_iters=T_max, full_trace=True, Ts=None, tol=tol
)

print("Backward Pass Error (DEER vs Manual):", jnp.max(jnp.abs(costate_deer - lambda_traj_rev)))




# Flip back to get chronological order: [lambda_0, ..., lambda_{T-1}]
lambda_traj = jnp.flip(lambda_traj_rev, axis=0)

# --- Compute Policy Gradient (\nabla_K J) ---
def compute_grad_step(carry, inputs):
    x_k, lambda_k_plus_1 = inputs
    u_k = -K @ x_k
    
    # 2R u_k + g(x_k)^T \lambda_{k+1} (Assuming R=I, g(x_k) = B)
    h_u = 2 * u_k + B.T @ lambda_k_plus_1
    
    # \nabla_K J step = - (h_u) @ x_k^T
    grad_K_step = -jnp.outer(h_u, x_k)
    return carry, grad_K_step

# Note: lambda_traj contains lambda_0 to lambda_{T-1}. 
# For lambda_{k+1}, we shift the trajectory by 1 and append lambda_T
lambda_k_plus_1_traj = jnp.vstack([lambda_traj[1:], lambda_T])

_, all_K_grads = jax.lax.scan(compute_grad_step, None, (x_traj, lambda_k_plus_1_traj))

# Sum gradients over all time steps
grad_K = jnp.sum(all_K_grads, axis=0)
print("\nGradient w.r.t K:\n", grad_K)

# --- Gradient Update ---
learning_rate = 1e-4
K_new = K - learning_rate * grad_K
print("\nUpdated Feedback Matrix K:\n", K_new)