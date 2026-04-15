"""
src/cuda_policy_iteration.py — Generic 2D CUDA Policy Iteration base class.

Mirrors the PolicyIterationStall architecture (stall-spin-recovery-dp) for
2-dimensional state spaces. Subclasses inject environment dynamics as a CUDA
__device__ function named `step_dynamics`, keeping the eval/improve kernels
completely generic.

Interface for subclasses:
  - Override `_dynamics_cuda_src()` → return CUDA source string containing:
        __device__ void step_dynamics(
            float s0, float s1, float action,
            float* next_s0, float* next_s1,
            float* reward, bool* terminated
        )
  - Override `_terminal_fn(states)` → return (bool_mask, terminal_value)
    for environments that have absorbing terminal states (default: none).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class CudaPIConfig:
    """Configuration for CudaPolicyIteration2D."""
    gamma: float = 0.99           # Discount factor
    theta: float = 1e-4           # Convergence threshold (policy evaluation)
    max_eval_iter: int = 10_000   # Max iterations per policy evaluation sweep
    max_pi_iter: int = 50         # Max outer policy iteration loops
    log_interval: int = 100       # Log every N evaluation iterations


class CudaPolicyIteration2D(abc.ABC):
    """
    High-performance CUDA Policy Iteration for 2D continuous state spaces.

    State space is discretized on a regular 2D grid. Dynamics are embedded
    directly in CUDA kernels (no Gymnasium env calls during DP iterations).
    Barycentric interpolation (4 corners, 2D) maps continuous next-states
    back to grid values.

    Parameters
    ----------
    bins_space   : dict with two keys mapping to 1D linspace arrays.
                   e.g. {"position": np.linspace(-1.2, 0.6, 200),
                          "velocity": np.linspace(-0.07, 0.07, 200)}
    action_space : 1D np.ndarray of scalar action values.
                   e.g. np.array([-1.0, 0.0, 1.0])
    config       : CudaPIConfig instance.
    """

    def __init__(
        self,
        bins_space: dict,
        action_space: np.ndarray,
        config: CudaPIConfig | None = None,
    ) -> None:
        if not GPU_AVAILABLE:
            raise RuntimeError(
                "CuPy is required for CudaPolicyIteration2D. "
                "Install with: pip install cupy-cuda12x  (or matching CUDA version)"
            )

        self.config = config or CudaPIConfig()
        self.action_space = np.ascontiguousarray(action_space, dtype=np.float32)
        self.n_actions = len(self.action_space)

        # Build flat states_space: (n_states, 2)
        keys = list(bins_space.keys())
        assert len(keys) == 2, "CudaPolicyIteration2D requires exactly 2 state dimensions."
        grids = np.meshgrid(bins_space[keys[0]], bins_space[keys[1]], indexing="ij")
        self.states_space = np.column_stack(
            [g.ravel() for g in grids]
        ).astype(np.float32)
        self.n_states = len(self.states_space)

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

    # ── Grid metadata ─────────────────────────────────────────────────────────

    def _precompute_grid_metadata(self) -> None:
        self.bounds_low = np.min(self.states_space, axis=0).astype(np.float32)
        self.bounds_high = np.max(self.states_space, axis=0).astype(np.float32)

        self.grid_shape = np.array(
            [len(np.unique(self.states_space[:, d])) for d in range(2)],
            dtype=np.int32,
        )
        # Row-major strides: strides[0]=n_cols, strides[1]=1
        self.strides = np.array([self.grid_shape[1], 1], dtype=np.int32)
        self.corner_bits = np.array(list(product([0, 1], repeat=2)), dtype=np.int32)
        logger.info(
            f"Grid: shape={self.grid_shape.tolist()}, "
            f"states={self.n_states:,}, actions={self.n_actions}"
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def _dynamics_cuda_src(self) -> str:
        """
        Return CUDA source string containing the step_dynamics __device__ function.

        Must implement:
            __device__ void step_dynamics(
                float s0, float s1, float action,
                float* next_s0, float* next_s1,
                float* reward, bool* terminated
            )
        """
        ...

    def _terminal_fn(self, states: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Identify absorbing terminal states and their initial value.

        Override for environments with goal/crash states.
        Default: no terminal states (Pendulum-style).

        Returns
        -------
        (bool_mask of shape (n_states,), scalar terminal value)
        """
        return np.zeros(len(states), dtype=bool), 0.0

    # ── GPU allocation & compilation ──────────────────────────────────────────

    def _allocate_tensors_and_compile(self) -> None:
        logger.info("Allocating GPU tensors and compiling CUDA kernels...")

        self.d_states = cp.asarray(self.states_space, dtype=cp.float32)
        self.d_actions = cp.asarray(self.action_space, dtype=cp.float32)
        self.d_bounds_low = cp.asarray(self.bounds_low, dtype=cp.float32)
        self.d_bounds_high = cp.asarray(self.bounds_high, dtype=cp.float32)
        self.d_grid_shape = cp.asarray(self.grid_shape, dtype=cp.int32)
        self.d_strides = cp.asarray(self.strides, dtype=cp.int32)

        self.d_policy = cp.zeros(self.n_states, dtype=cp.int32)
        self.d_value_function = cp.zeros(self.n_states, dtype=cp.float32)
        self.d_new_value_function = cp.zeros(self.n_states, dtype=cp.float32)

        terminal_mask, terminal_value = self._terminal_fn(self.states_space)
        self.d_terminal_mask = cp.asarray(terminal_mask, dtype=cp.bool_)
        if np.any(terminal_mask):
            self.d_value_function[self.d_terminal_mask] = float(terminal_value)
            logger.info(f"Terminal states: {terminal_mask.sum():,} (value={terminal_value})")
        self.d_new_value_function[:] = self.d_value_function[:]

        # Fused reduction kernel: max(|A - B|) with no auxiliary VRAM
        self._max_abs_diff = cp.ReductionKernel(
            in_params="float32 x, float32 y",
            out_params="float32 z",
            map_expr="abs(x - y)",
            reduce_expr="max(a, b)",
            post_map_expr="z = a",
            identity="0.0f",
            name="max_abs_diff_2d_pi",
        )

        self._compile_cuda_module()
        logger.success("CUDA kernels compiled. VRAM allocated.")

    def _compile_cuda_module(self) -> None:
        # Barycentric 2D interpolation + generic eval/improve kernels.
        # Subclass dynamics are prepended before the extern "C" block.
        generic_kernels = r'''
        extern "C" {

        __device__ void get_barycentric_2d(
            float s0, float s1,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int* idxs, float* wgts
        ) {
            float n0 = (s0 - b_low[0]) / (b_high[0] - b_low[0]) * (float)(g_shape[0] - 1);
            float n1 = (s1 - b_low[1]) / (b_high[1] - b_low[1]) * (float)(g_shape[1] - 1);

            n0 = fmaxf(0.0f, fminf(n0, (float)(g_shape[0] - 1)));
            n1 = fmaxf(0.0f, fminf(n1, (float)(g_shape[1] - 1)));

            int i0 = min((int)n0, g_shape[0] - 2);
            int i1 = min((int)n1, g_shape[1] - 2);

            float d0 = n0 - (float)i0;
            float d1 = n1 - (float)i1;

            idxs[0] =  i0      * strides[0] +  i1      * strides[1];
            idxs[1] =  i0      * strides[0] + (i1 + 1) * strides[1];
            idxs[2] = (i0 + 1) * strides[0] +  i1      * strides[1];
            idxs[3] = (i0 + 1) * strides[0] + (i1 + 1) * strides[1];

            wgts[0] = (1.0f - d0) * (1.0f - d1);
            wgts[1] = (1.0f - d0) * d1;
            wgts[2] = d0 * (1.0f - d1);
            wgts[3] = d0 * d1;
        }

        __global__ void policy_eval_kernel(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int n_states, float gamma_discount
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) { new_V[s_idx] = V[s_idx]; return; }

            float s0     = states[s_idx * 2 + 0];
            float s1     = states[s_idx * 2 + 1];
            float action = actions[policy[s_idx]];

            float next_s0, next_s1, reward;
            bool terminated;
            step_dynamics(s0, s1, action, &next_s0, &next_s1, &reward, &terminated);

            float expected_v = 0.0f;
            if (!terminated) {
                int idxs[4]; float wgts[4];
                get_barycentric_2d(next_s0, next_s1, b_low, b_high,
                                   g_shape, strides, idxs, wgts);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
                }
            }
            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        __global__ void policy_improve_kernel(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* b_low, const float* b_high,
            const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float s0 = states[s_idx * 2 + 0];
            float s1 = states[s_idx * 2 + 1];

            float max_q = -1.0e30f;
            int best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float action = actions[a];
                float next_s0, next_s1, reward;
                bool terminated;
                step_dynamics(s0, s1, action, &next_s0, &next_s1, &reward, &terminated);

                float expected_v = 0.0f;
                if (!terminated) {
                    int idxs[4]; float wgts[4];
                    get_barycentric_2d(next_s0, next_s1, b_low, b_high,
                                       g_shape, strides, idxs, wgts);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
                    }
                }

                float q_val = reward + gamma_discount * expected_v;
                if (q_val > max_q) { max_q = q_val; best_a = a; }
            }

            policy[s_idx] = best_a;
        }

        } // extern "C"
        '''

        cuda_source = self._dynamics_cuda_src() + generic_kernels
        module = cp.RawModule(code=cuda_source)
        self.eval_kernel = module.get_function("policy_eval_kernel")
        self.improve_kernel = module.get_function("policy_improve_kernel")

        self.threads_per_block = 256
        self.blocks_per_grid = (
            self.n_states + self.threads_per_block - 1
        ) // self.threads_per_block

    # ── Policy Iteration ──────────────────────────────────────────────────────

    def policy_evaluation(self) -> float:
        """Runs iterative policy evaluation on GPU until convergence."""
        delta = float("inf")
        SYNC_INTERVAL = 25

        for i in range(self.config.max_eval_iter):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function,
                    self.d_terminal_mask,
                    self.d_bounds_low, self.d_bounds_high,
                    self.d_grid_shape, self.d_strides,
                    np.int32(self.n_states), np.float32(self.config.gamma),
                )
            )

            d_delta = self._max_abs_diff(
                self.d_new_value_function, self.d_value_function
            )
            self.d_value_function, self.d_new_value_function = (
                self.d_new_value_function, self.d_value_function
            )

            if i % SYNC_INTERVAL == 0 or i == self.config.max_eval_iter - 1:
                delta = float(d_delta.get())
                if i % self.config.log_interval == 0:
                    logger.debug(f"  Eval iter {i:5d} | Δ = {delta:.4e}")
                if delta < self.config.theta:
                    logger.success(f"  Eval converged at iter {i} | Δ = {delta:.2e}")
                    return delta

        logger.warning(
            f"  Eval hit max_eval_iter={self.config.max_eval_iter} | Δ = {delta:.2e}"
        )
        return delta

    def policy_improvement(self) -> bool:
        """Greedy policy improvement on GPU. Returns True if policy is stable."""
        old_policy = self.d_policy.copy()

        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_bounds_low, self.d_bounds_high,
                self.d_grid_shape, self.d_strides,
                np.int32(self.n_states), np.int32(self.n_actions),
                np.float32(self.config.gamma),
            )
        )

        stable = bool(cp.all(self.d_policy == old_policy))
        return stable

    def run(self) -> None:
        """Execute the complete Policy Iteration loop."""
        for n in range(self.config.max_pi_iter):
            logger.info(f"── PI Iteration {n + 1}/{self.config.max_pi_iter} ──")
            self.policy_evaluation()
            if self.policy_improvement():
                logger.success(f"Policy Iteration converged at iteration {n + 1}.")
                break
        else:
            logger.warning(
                f"Policy Iteration hit max_pi_iter={self.config.max_pi_iter}."
            )

        self._pull_tensors_from_gpu()

    def _pull_tensors_from_gpu(self) -> None:
        """Transfer converged policy and value function to CPU RAM."""
        logger.info("Pulling results from VRAM to RAM...")

        self.value_function: np.ndarray = self.d_value_function.get()
        self.policy: np.ndarray = self.d_policy.get()

        for attr in [
            "d_states", "d_actions", "d_bounds_low", "d_bounds_high",
            "d_grid_shape", "d_strides", "d_terminal_mask",
            "d_value_function", "d_new_value_function", "d_policy",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

        cp.get_default_memory_pool().free_all_blocks()
        logger.success("VRAM released. Results in CPU RAM.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: Path | str) -> None:
        """Serialize policy and value function to a .npz archive."""
        filepath = Path(filepath).with_suffix(".npz")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            filepath,
            value_function=self.value_function,
            policy=self.policy,
            bounds_low=self.bounds_low,
            bounds_high=self.bounds_high,
            grid_shape=self.grid_shape,
            strides=self.strides,
            corner_bits=self.corner_bits,
            action_space=self.action_space,
            states_space=self.states_space,
        )
        logger.success(f"Policy saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path | str) -> "CudaPolicyIteration2D":
        """Load a saved policy from a .npz archive (no GPU required)."""
        filepath = Path(filepath).with_suffix(".npz")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.value_function = data["value_function"]
        instance.policy = data["policy"]
        instance.bounds_low = data["bounds_low"]
        instance.bounds_high = data["bounds_high"]
        instance.grid_shape = data["grid_shape"]
        instance.strides = data["strides"]
        instance.corner_bits = data["corner_bits"]
        instance.action_space = data["action_space"]
        instance.states_space = data["states_space"]
        instance.n_actions = len(instance.action_space)
        instance.n_states = len(instance.states_space)
        instance.config = CudaPIConfig()

        logger.success(f"Policy loaded from {filepath.resolve()}")
        return instance
