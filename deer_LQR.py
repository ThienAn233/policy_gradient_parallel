"""
deer_fixed_j.py

Fixed-Jacobian DEER implementation.

This is a drop-in-style DEER variant for systems where the Jacobian
with respect to the trajectory variable is known and fixed.

For LQR / linear closed-loop dynamics,

    x_{k+1} = (A - B K) x_k,

the fixed Jacobian is

    J = A - B K.

For the backward costate recursion,

    lambda_k = grad_x l(x_k, K) + (A - B K)^T lambda_{k+1},

the fixed Jacobian with respect to lambda_{k+1} is

    J = (A - B K)^T.

This avoids the expensive call

    jax.jacrev(f, argnums=0)

inside the original DEER implementation.
"""

import jax
jax.config.update("jax_enable_x64", True)

from jax import vmap
from jax.lax import scan
import jax.numpy as jnp


# ============================================================
# Associative operator for affine recurrences
# ============================================================

def full_mat_operator_fixed_j(q_i, q_j):
    """
    Binary operator for affine map composition.

    q_i = (A_i, b_i) represents
        z_out = A_i z_in + b_i.

    q_j = (A_j, b_j) represents
        z_out = A_j z_in + b_j.

    q_j o q_i is
        z_out = A_j A_i z_in + A_j b_i + b_j.

    Works with leading batch dimensions, so it can be used directly
    inside jax.lax.associative_scan.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    A_out = jnp.matmul(A_j, A_i)
    b_out = jnp.einsum("...ij,...j->...i", A_j, b_i) + b_j

    return A_out, b_out


# ============================================================
# Residual and merit function
# ============================================================

def get_residual_fixed_j(f, initial_state, states, drivers):
    """
    Residual for trajectory equation

        states[k] = f(previous_state, drivers[k])

    where states stores [z_1, ..., z_T].
    """
    fs_inner = vmap(f)(states[:-1], drivers[1:])
    f0 = f(initial_state, drivers[0])

    fs = jnp.concatenate([f0[jnp.newaxis, :], fs_inner], axis=0)

    return states - fs


def merit_fxn_fixed_j(f, initial_state, states, drivers, Ts=None):
    """
    Average residual merit:

        L = 1/(2T) sum_k ||r_k||^2.

    If Ts is provided, returns the merit value for multiple truncation
    horizons.
    """
    r = get_residual_fixed_j(f, initial_state, states, drivers)
    T = r.shape[0]

    cumulative = 0.5 * jnp.sum(jnp.cumsum(r**2, axis=0), axis=1)
    cumulative = jnp.where(jnp.isnan(cumulative), jnp.inf, cumulative)

    if Ts is not None:
        return cumulative[Ts - 1] / Ts
    return cumulative[-1] / T


# ============================================================
# Fixed-J DEER
# ============================================================

def deer_alg_fixed_j(
    f,
    J,
    initial_state,
    states_guess,
    drivers,
    num_iters,
    k=0.0,
    full_trace=False,
    Ts=None,
    tol=1e-10,
):
    """
    Fixed-Jacobian DEER.

    Args:
        f:
            Function of the form

                z_next = f(z, driver)

            where z is the optimized trajectory variable and driver is
            fixed external data.

        J:
            Fixed Jacobian df/dz, shape (D, D).

            For forward LQR:
                J = A - B @ K.

            For backward LQR costate:
                J = (A - B @ K).T.

            For one-pass augmented decoupled LQR:
                J = block_diag(F_cl, F_cl.T).

        initial_state:
            Fixed initial value z_0, shape (D,).

        states_guess:
            Initial guess for [z_1, ..., z_T], shape (T, D).

        drivers:
            Driver sequence, shape (T, driver_dim).

        num_iters:
            Maximum number of DEER iterations.

        k:
            Damping parameter. Uses J_eff = (1-k) J.

        full_trace:
            If True, returns all iterates.

        Ts:
            Optional array of truncation horizons.

        tol:
            Stop when merit is below tol.

    Returns:
        Same style as the original deer_alg:

        (
            all_states,
            final_state,
            newton_steps,
            lle,
            is_nan,
            nan_mask,
            all_newtons,
            mf_val,
        )

        Here lle, nan_mask, all_newtons are returned as None.
    """
    T = states_guess.shape[0]
    D = initial_state.shape[0]

    J_eff = (1.0 - k) * J

    @jax.jit
    def _step(carry, args):
        states, is_nan = carry

        # Evaluate f in parallel on inner timesteps:
        # states[:-1] = [z_1, ..., z_{T-1}]
        # drivers[1:] = [d_1, ..., d_{T-1}]
        fs_inner = vmap(f)(states[:-1], drivers[1:])

        # For affine approximation:
        #     f(z, d) approx J_eff z + b(d)
        # so:
        #     b(d) = f(z, d) - J_eff z.
        b_inner = fs_inner - jnp.einsum("ij,tj->ti", J_eff, states[:-1])

        # First step uses fixed initial_state z_0:
        #     z_1 = f(z_0, d_0)
        # This is represented by A_0 = 0 and b_0 = f(z_0, d_0).
        b0 = f(initial_state, drivers[0])
        A0 = jnp.zeros_like(J_eff)

        A_inner = jnp.broadcast_to(J_eff, (T - 1, D, D))

        A_seq = jnp.concatenate([A0[jnp.newaxis, :, :], A_inner], axis=0)
        b_seq = jnp.concatenate([b0[jnp.newaxis, :], b_inner], axis=0)

        _, new_states = jax.lax.associative_scan(
            full_mat_operator_fixed_j,
            (A_seq, b_seq),
        )

        is_nan = jnp.logical_or(is_nan, jnp.isnan(new_states).any())
        mf_val = merit_fxn_fixed_j(f, initial_state, new_states, drivers)

        return (new_states, is_nan), (new_states, mf_val)

    def cond_func(iter_inp):
        iter_idx, _, err, *_ = iter_inp
        return jnp.logical_and(iter_idx < num_iters, err > tol)

    def body_func_single(iter_inp):
        iter_idx, states, _, is_nan, _ = iter_inp
        step_output, step_aux = _step((states, is_nan), None)
        new_states, new_is_nan = step_output
        _, new_err = step_aux
        return iter_idx + 1, new_states, new_err, new_is_nan, None

    def body_func_multiple(iter_inp):
        iter_idx, states, _, is_nan, iters_below = iter_inp
        step_output, step_aux = _step((states, is_nan), None)
        new_states, new_is_nan = step_output

        new_errs = merit_fxn_fixed_j(
            f,
            initial_state,
            new_states,
            drivers,
            Ts=Ts,
        )

        iters_below = iters_below + (new_errs < tol).astype(int)

        return iter_idx + 1, new_states, new_errs[-1], new_is_nan, iters_below

    if full_trace:
        last_output, all_outputs = scan(
            _step,
            (states_guess, False),
            None,
            length=num_iters,
        )

        final_state, is_nan = last_output
        all_states, mf_vals = all_outputs

        all_states = jnp.concatenate([states_guess[jnp.newaxis, ...], all_states])
        mf_val = mf_vals[-1]
        newton_steps = None
        all_newtons = None
        nan_mask = None

    elif Ts is not None:
        newton_steps, final_state, final_err, is_nan, iters_below = jax.lax.while_loop(
            cond_func,
            body_func_multiple,
            (
                0,
                states_guess,
                merit_fxn_fixed_j(f, initial_state, states_guess, drivers),
                False,
                jnp.zeros_like(Ts),
            ),
        )

        all_states = None
        mf_val = final_err
        nan_mask = None
        all_newtons = newton_steps - iters_below + 1

    else:
        newton_steps, final_state, mf_val, is_nan, _ = jax.lax.while_loop(
            cond_func,
            body_func_single,
            (
                0,
                states_guess,
                merit_fxn_fixed_j(f, initial_state, states_guess, drivers),
                False,
                None,
            ),
        )

        all_states = None
        nan_mask = None
        all_newtons = None

    lle = None

    return (
        all_states,
        final_state,
        newton_steps,
        lle,
        is_nan,
        nan_mask,
        all_newtons,
        mf_val,
    )