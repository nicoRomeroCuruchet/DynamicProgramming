try:
    from utils.barycentric import get_barycentric_weights_and_indices, get_optimal_action
except ImportError:
    pass  # numba not available
