"""
utils/barycentric.py — O(1) Barycentric interpolation for regular N-dimensional grids.

Ported from the stall-spin-recovery-dp project (utils/utils.py).
Uses numba JIT compilation for maximum CPU performance.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def get_barycentric_weights_and_indices(
    points: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    grid_shape: np.ndarray,
    strides: np.ndarray,
    corner_bits: np.ndarray,
) -> tuple:
    """
    Compute O(1) barycentric interpolation weights and flat grid indices
    for a batch of points on a regular N-dimensional grid.

    Parameters
    ----------
    points      : (n_points, n_dims) float32 — query points
    bounds_low  : (n_dims,)          float32 — grid lower bounds per dimension
    bounds_high : (n_dims,)          float32 — grid upper bounds per dimension
    grid_shape  : (n_dims,)          int32   — number of bins per dimension
    strides     : (n_dims,)          int32   — flat-index strides per dimension
    corner_bits : (2**n_dims, n_dims) int32  — hypercube corner offsets {0,1}^n_dims

    Returns
    -------
    weights : (n_points, 2**n_dims) float32 — barycentric weights summing to 1
    indices : (n_points, 2**n_dims) int32   — flat indices into the value function
    """
    n_points, n_dims = points.shape
    n_corners = corner_bits.shape[0]

    weights = np.zeros((n_points, n_corners), dtype=np.float32)
    indices = np.zeros((n_points, n_corners), dtype=np.int32)

    step_sizes = (bounds_high - bounds_low) / (grid_shape - 1)

    for i in range(n_points):
        base_idx = np.zeros(n_dims, dtype=np.int32)
        t = np.zeros(n_dims, dtype=np.float32)

        for d in range(n_dims):
            p = max(bounds_low[d], min(points[i, d], bounds_high[d]))

            cell = (p - bounds_low[d]) / step_sizes[d]
            idx_d = int(cell)

            if idx_d >= grid_shape[d] - 1:
                idx_d = grid_shape[d] - 2

            base_idx[d] = idx_d
            t[d] = (p - (bounds_low[d] + idx_d * step_sizes[d])) / step_sizes[d]

        for c in range(n_corners):
            w = 1.0
            flat_idx = 0
            for d in range(n_dims):
                bit = corner_bits[c, d]
                w *= t[d] if bit else (1.0 - t[d])
                flat_idx += (base_idx[d] + bit) * strides[d]

            weights[i, c] = w
            indices[i, c] = flat_idx

    return weights, indices


def get_optimal_action(
    state: np.ndarray,
    policy: np.ndarray,
    action_space: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    grid_shape: np.ndarray,
    strides: np.ndarray,
    corner_bits: np.ndarray,
) -> np.ndarray:
    """
    Interpolate the optimal action for a given continuous state.

    Uses barycentric interpolation over the surrounding grid vertices
    to produce a smooth action from the discrete policy table.

    Returns
    -------
    np.ndarray — interpolated optimal action
    """
    state_2d = np.atleast_2d(state).astype(np.float32)

    lambdas, flat_indices = get_barycentric_weights_and_indices(
        state_2d, bounds_low, bounds_high, grid_shape, strides, corner_bits
    )

    lambdas = lambdas.flatten()
    flat_indices = flat_indices.flatten()

    best_action_indices = policy[flat_indices]
    neighbor_actions = action_space[best_action_indices]

    return lambdas @ neighbor_actions
