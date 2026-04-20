"""
src/adaptive_cuda_pi.py — Hessian-based Adaptive Simplicial Mesh for 2D CUDA Policy Iteration.

Implements Definition 3.2 of the article:
    δ(ξ) ≤ sqrt(ε / (‖∇²V(ξ)‖ + ν))

The tensor-product grid (hipercubo) is kept as the mesh structure — no Delaunay needed.
Adaptivity comes from placing nodes non-uniformly: denser where ‖∇²V‖ is large.

Changes vs CudaPolicyIteration2D:
  - Barycentric lookup: O(1) division → O(log n) binary search per dim
  - Adds: Hessian estimation kernel, equidistribution (CPU), value transfer
  - Adds: run_adaptive() loop — coarse solve → hessian → equidistribute → transfer → fine solve
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.interpolate import RegularGridInterpolator

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from src.cuda_policy_iteration import CudaPolicyIteration2D, CudaPolicyIteration4D, CudaPIConfig


@dataclass
class AdaptivePIConfig:
    epsilon: float = 0.01   # interpolation error tolerance (drives node density)
    nu: float = 0.1         # regularization — prevents zero cell size where ‖∇²V‖≈0
    n_refine: int = 2       # number of mesh refinement cycles
    n_nodes_refined: int = 300  # nodes per dimension after each refinement


class AdaptiveCudaPI2D(CudaPolicyIteration2D):
    """
    Hessian-based adaptive mesh Policy Iteration for 2D continuous state spaces.

    Inherits the full eval/improve architecture from CudaPolicyIteration2D.
    Replaces the O(1) uniform barycentric lookup with an O(log n) binary search
    over explicitly stored per-dimension node coordinate arrays.

    Subclass interface (same as CudaPolicyIteration2D):
      - Override _dynamics_cuda_src()  → step_dynamics __device__ function
      - Override _terminal_fn(states)  → (bool_mask, terminal_value)
    """

    def __init__(
        self,
        bins_space: dict,
        action_space: np.ndarray,
        config: CudaPIConfig | None = None,
        adaptive_config: AdaptivePIConfig | None = None,
    ) -> None:
        self.adaptive_config = adaptive_config or AdaptivePIConfig()
        # Store explicit node coordinates per dimension.
        # Must be set BEFORE super().__init__ because the overridden
        # _precompute_grid_metadata and _allocate_tensors_and_compile
        # are called from there.
        keys = list(bins_space.keys())
        assert len(keys) == 2
        self.coords_per_dim = [bins_space[keys[0]].copy().astype(np.float32),
                               bins_space[keys[1]].copy().astype(np.float32)]
        self.dim_names = keys
        super().__init__(bins_space, action_space, config)

    # ── Grid metadata (override) ───────────────────────────────────────────────

    def _precompute_grid_metadata(self) -> None:
        """Use explicit coords_per_dim instead of inferring from states_space."""
        self.bounds_low  = np.array([c[0]  for c in self.coords_per_dim], dtype=np.float32)
        self.bounds_high = np.array([c[-1] for c in self.coords_per_dim], dtype=np.float32)
        self.grid_shape  = np.array([len(c) for c in self.coords_per_dim], dtype=np.int32)
        self.strides     = np.array([self.grid_shape[1], 1], dtype=np.int32)
        # corner_bits kept for CPU-side barycentric (get_optimal_action compatibility)
        from itertools import product as iproduct
        self.corner_bits = np.array(list(iproduct([0, 1], repeat=2)), dtype=np.int32)
        logger.info(
            f"Adaptive grid: shape={self.grid_shape.tolist()}, "
            f"states={self.n_states:,}, actions={self.n_actions}"
        )

    # ── GPU allocation (override) ──────────────────────────────────────────────

    def _allocate_tensors_and_compile(self) -> None:
        """Extend parent: add per-dim coordinate arrays on GPU."""
        super()._allocate_tensors_and_compile()
        self.d_coords0 = cp.asarray(self.coords_per_dim[0], dtype=cp.float32)
        self.d_coords1 = cp.asarray(self.coords_per_dim[1], dtype=cp.float32)

    # ── CUDA module (override) ─────────────────────────────────────────────────

    def _compile_cuda_module(self) -> None:
        """Compile adaptive kernels: binary-search barycentric + Hessian estimator."""
        adaptive_kernels = r'''
        extern "C" {

        // Binary search: returns index i s.t. coords[i] <= val < coords[i+1].
        // Clamps to [0, n-2].
        __device__ int find_cell(const float* coords, int n, float val) {
            if (val <= coords[0])   return 0;
            if (val >= coords[n-1]) return n - 2;
            int lo = 0, hi = n - 2;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (coords[mid] <= val) lo = mid;
                else                    hi = mid - 1;
            }
            return lo;
        }

        // Adaptive barycentric: O(log n) per dimension.
        __device__ void get_barycentric_2d_adaptive(
            float s0, float s1,
            const float* coords0, const float* coords1,
            int n0, int n1, const int* strides,
            int* idxs, float* wgts
        ) {
            int i0 = find_cell(coords0, n0, s0);
            int i1 = find_cell(coords1, n1, s1);

            float t0 = (s0 - coords0[i0]) / (coords0[i0 + 1] - coords0[i0]);
            float t1 = (s1 - coords1[i1]) / (coords1[i1 + 1] - coords1[i1]);
            t0 = fmaxf(0.0f, fminf(1.0f, t0));
            t1 = fmaxf(0.0f, fminf(1.0f, t1));

            idxs[0] =  i0      * strides[0] +  i1;      wgts[0] = (1.0f - t0) * (1.0f - t1);
            idxs[1] =  i0      * strides[0] + (i1 + 1); wgts[1] = (1.0f - t0) * t1;
            idxs[2] = (i0 + 1) * strides[0] +  i1;      wgts[2] = t0 * (1.0f - t1);
            idxs[3] = (i0 + 1) * strides[0] + (i1 + 1); wgts[3] = t0 * t1;
        }

        // Frobenius norm of the 2x2 Hessian of V, estimated via finite differences.
        // Uses non-uniform spacing: hl = left spacing, hr = right spacing.
        // Boundary nodes are set to 0 (one-sided differences not implemented).
        __global__ void compute_hessian_norm(
            const float* V,
            const float* coords0, const float* coords1,
            float* hess_norm,
            int n0, int n1
        ) {
            int s = blockIdx.x * blockDim.x + threadIdx.x;
            if (s >= n0 * n1) return;
            int i = s / n1;
            int j = s % n1;

            if (i == 0 || i == n0 - 1 || j == 0 || j == n1 - 1) {
                hess_norm[s] = 0.0f;
                return;
            }

            float hl0 = coords0[i]     - coords0[i - 1];
            float hr0 = coords0[i + 1] - coords0[i];
            float hl1 = coords1[j]     - coords1[j - 1];
            float hr1 = coords1[j + 1] - coords1[j];

            // Non-uniform 2nd derivative (exact for quadratics)
            float d2_dx2 = 2.0f * (
                  V[(i + 1) * n1 + j] / hr0
                - V[ i      * n1 + j] * (hl0 + hr0) / (hl0 * hr0)
                + V[(i - 1) * n1 + j] / hl0
            ) / (hl0 + hr0);

            float d2_dy2 = 2.0f * (
                  V[i * n1 + (j + 1)] / hr1
                - V[i * n1 +  j     ] * (hl1 + hr1) / (hl1 * hr1)
                + V[i * n1 + (j - 1)] / hl1
            ) / (hl1 + hr1);

            // Mixed derivative - 4-point central difference
            float dx = 0.5f * (hl0 + hr0);
            float dy = 0.5f * (hl1 + hr1);
            float d2_dxdy = (
                  V[(i + 1) * n1 + (j + 1)]
                - V[(i + 1) * n1 + (j - 1)]
                - V[(i - 1) * n1 + (j + 1)]
                + V[(i - 1) * n1 + (j - 1)]
            ) / (4.0f * dx * dy);

            // Frobenius norm: ||H||_F = sqrt(H00^2 + 2*H01^2 + H11^2)
            hess_norm[s] = sqrtf(
                d2_dx2 * d2_dx2 + 2.0f * d2_dxdy * d2_dxdy + d2_dy2 * d2_dy2
            );
        }

        __global__ void policy_eval_kernel_adaptive(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* coords0, const float* coords1,
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
                get_barycentric_2d_adaptive(
                    next_s0, next_s1,
                    coords0, coords1,
                    g_shape[0], g_shape[1], strides,
                    idxs, wgts
                );
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
            }
            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        __global__ void policy_improve_kernel_adaptive(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* coords0, const float* coords1,
            const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float s0 = states[s_idx * 2 + 0];
            float s1 = states[s_idx * 2 + 1];

            float max_q = -1.0e30f;
            int   best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float action = actions[a];
                float next_s0, next_s1, reward;
                bool terminated;
                step_dynamics(s0, s1, action, &next_s0, &next_s1, &reward, &terminated);

                float expected_v = 0.0f;
                if (!terminated) {
                    int idxs[4]; float wgts[4];
                    get_barycentric_2d_adaptive(
                        next_s0, next_s1,
                        coords0, coords1,
                        g_shape[0], g_shape[1], strides,
                        idxs, wgts
                    );
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
                }

                float q_val = reward + gamma_discount * expected_v;
                if (q_val > max_q) { max_q = q_val; best_a = a; }
            }
            policy[s_idx] = best_a;
        }

        } // extern "C"
        '''

        cuda_source = self._dynamics_cuda_src() + adaptive_kernels
        module = cp.RawModule(code=cuda_source)

        self.hessian_kernel = module.get_function("compute_hessian_norm")
        self.eval_kernel    = module.get_function("policy_eval_kernel_adaptive")
        self.improve_kernel = module.get_function("policy_improve_kernel_adaptive")

        self.threads_per_block = 256
        self.blocks_per_grid = (
            self.n_states + self.threads_per_block - 1
        ) // self.threads_per_block

    # ── Policy iteration overrides (pass coord arrays) ─────────────────────────

    def policy_evaluation(self) -> float:
        delta = float("inf")
        SYNC_INTERVAL = 25

        for i in range(self.config.max_eval_iter):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function,
                    self.d_terminal_mask,
                    self.d_coords0, self.d_coords1,
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
        old_policy = self.d_policy.copy()
        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_coords0, self.d_coords1,
                self.d_grid_shape, self.d_strides,
                np.int32(self.n_states), np.int32(self.n_actions),
                np.float32(self.config.gamma),
            )
        )
        return bool(cp.all(self.d_policy == old_policy))

    # ── Inner PI loop (no GPU teardown) ───────────────────────────────────────

    def _run_pi_loop(self) -> None:
        """Run PI iterations without pulling results from GPU."""
        for n in range(self.config.max_pi_iter):
            logger.info(f"── PI Iteration {n + 1}/{self.config.max_pi_iter} ──")
            self.policy_evaluation()
            if self.policy_improvement():
                logger.success(f"PI converged at iteration {n + 1}.")
                return
        logger.warning(f"PI hit max_pi_iter={self.config.max_pi_iter}.")

    # ── Adaptive mesh steps ────────────────────────────────────────────────────

    def compute_hessian_norms(self) -> np.ndarray:
        """
        Estimate ‖∇²V‖ (Frobenius) at every grid node via GPU finite differences.
        Returns CPU array of shape (n_states,).
        """
        d_hess = cp.zeros(self.n_states, dtype=cp.float32)
        n0, n1 = int(self.grid_shape[0]), int(self.grid_shape[1])

        self.hessian_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_value_function,
                self.d_coords0, self.d_coords1,
                d_hess,
                np.int32(n0), np.int32(n1),
            )
        )
        result = d_hess.get()
        d_hess = None
        cp.get_default_memory_pool().free_all_blocks()
        return result

    def equidistribute_nodes(
        self, hess_norm: np.ndarray, n_new: int
    ) -> list[np.ndarray]:
        """
        Place n_new nodes per dimension using the equidistribution principle.

        For each dimension d, the 1D density is:
            ρ_d(x) = sqrt((max_{other dims} ‖∇²V‖(x) + ν) / ε)

        Nodes are placed so that each interval carries equal "arc length"
        in the density-weighted space — more nodes where curvature is high.
        """
        cfg = self.adaptive_config
        hess_2d = hess_norm.reshape(self.grid_shape)
        new_coords = []

        for d in range(2):
            other = 1 - d
            # marginal curvature: max over the other dimension
            curvature_1d = hess_2d.max(axis=other)

            # Normalize to [0, 1] so the absolute scale of V doesn't bias
            # node placement. Without this, environments with large rewards
            # (e.g. +100 goal bonus) create a Hessian spike at the terminal
            # boundary that is orders of magnitude larger than the true
            # curvature structure, concentrating all nodes there.
            h_max = curvature_1d.max()
            if h_max > 0:
                curvature_1d = curvature_1d / h_max

            density = np.sqrt((curvature_1d + cfg.nu) / cfg.epsilon)
            density = np.maximum(density, 1e-8)

            # cumulative arc-length, normalized to [0, 1]
            cumulative = np.cumsum(density)
            cumulative /= cumulative[-1]

            old_coords = self.coords_per_dim[d]
            targets = np.linspace(0.0, 1.0, n_new)
            new_c = np.interp(targets, cumulative, old_coords).astype(np.float32)

            # Clamp endpoints to exact bounds
            new_c[0]  = old_coords[0]
            new_c[-1] = old_coords[-1]

            # Enforce strict monotonicity using np.nextafter (float32-safe).
            # np.interp can return duplicate values when the density has a spike:
            # many targets map to the same coordinate. Adding a fixed tiny delta
            # fails because it can be smaller than float32 epsilon at that
            # magnitude. np.nextafter always returns the next representable float.
            hi = np.float32(old_coords[-1] + 1.0)   # direction for nextafter
            for k in range(1, len(new_c) - 1):
                if new_c[k] <= new_c[k - 1]:
                    new_c[k] = np.nextafter(new_c[k - 1], hi)
            new_c[-1] = old_coords[-1]

            new_coords.append(new_c)

        return new_coords

    def transfer_value_function(
        self, new_coords: list[np.ndarray]
    ) -> np.ndarray:
        """
        Linearly interpolate the current V onto a new grid.
        Uses scipy RegularGridInterpolator (handles non-uniform grids).
        Returns CPU array of shape (n_new_states,).
        """
        V_2d = self.d_value_function.get().reshape(self.grid_shape)

        interp = RegularGridInterpolator(
            (self.coords_per_dim[0], self.coords_per_dim[1]),
            V_2d,
            method='linear',
            bounds_error=False,
            fill_value=None,   # extrapolate at boundaries
        )

        g0, g1 = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
        pts = np.stack([g0.ravel(), g1.ravel()], axis=-1)
        return interp(pts).astype(np.float32)

    def rebuild_grid(
        self, new_coords: list[np.ndarray], V_init: np.ndarray
    ) -> None:
        """
        Rebuild GPU state space with new non-uniform node coordinates.

        1. Frees all GPU arrays.
        2. Constructs new states_space from meshgrid of new_coords.
        3. Re-computes grid metadata, re-allocates GPU tensors, re-compiles kernels.
        4. Initializes d_value_function with the transferred V_init.
        """
        # Free GPU
        _gpu_attrs = [
            "d_states", "d_actions", "d_bounds_low", "d_bounds_high",
            "d_grid_shape", "d_strides", "d_terminal_mask",
            "d_value_function", "d_new_value_function", "d_policy",
            "d_coords0", "d_coords1",
        ]
        for attr in _gpu_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        cp.get_default_memory_pool().free_all_blocks()

        # Update grid
        self.coords_per_dim = new_coords
        g0, g1 = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
        self.states_space = np.column_stack(
            [g0.ravel(), g1.ravel()]
        ).astype(np.float32)
        self.n_states = len(self.states_space)

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()  # also sets d_coords0/1

        # Initialize V with the interpolated values
        self.d_value_function[:]     = cp.asarray(V_init, dtype=cp.float32)
        self.d_new_value_function[:] = self.d_value_function[:]

        # Re-apply terminal mask (it was set in _allocate_tensors_and_compile,
        # but overwritten by V_init above, so we re-enforce it)
        terminal_mask, terminal_value = self._terminal_fn(self.states_space)
        if np.any(terminal_mask):
            d_term = cp.asarray(terminal_mask, dtype=cp.bool_)
            self.d_value_function[d_term]     = float(terminal_value)
            self.d_new_value_function[d_term] = float(terminal_value)

        logger.info(
            f"Grid rebuilt → shape={self.grid_shape.tolist()}, "
            f"{self.n_states:,} states"
        )

    # ── Main adaptive loop ─────────────────────────────────────────────────────

    def run_adaptive(self) -> None:
        """
        Full adaptive mesh policy iteration (Definition 3.2):

            Phase 1: Coarse uniform solve.
            For each refinement cycle:
                Phase 2: Estimate ‖∇²V‖ on GPU (finite differences).
                Phase 3: Equidistribute nodes per dimension (CPU).
                Phase 4: Transfer V to new grid (CPU interpolation).
                Phase 5: Re-solve PI on the adaptive grid.
        """
        cfg = self.adaptive_config

        logger.info("══ Phase 1: Coarse uniform solve ══")
        self._run_pi_loop()

        for k in range(cfg.n_refine):
            logger.info(f"══ Refinement {k + 1}/{cfg.n_refine} ══")

            logger.info("  [2] Computing ‖∇²V‖ on GPU...")
            hess = self.compute_hessian_norms()
            logger.info(
                f"      ‖∇²V‖ — min={hess.min():.3f}  "
                f"max={hess.max():.3f}  mean={hess.mean():.3f}"
            )

            logger.info(f"  [3] Equidistributing {cfg.n_nodes_refined} nodes/dim...")
            new_coords = self.equidistribute_nodes(hess, cfg.n_nodes_refined)

            logger.info("  [4] Transferring V to new grid...")
            V_new = self.transfer_value_function(new_coords)

            logger.info("  [5] Rebuilding grid and re-solving...")
            self.rebuild_grid(new_coords, V_new)
            self._run_pi_loop()

        self._pull_tensors_from_gpu()
        logger.success("Adaptive mesh policy iteration complete.")

    # ── Persistence (override to save/load coords_per_dim) ────────────────────

    def save(self, filepath: Path | str) -> None:
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
            coords0=self.coords_per_dim[0],
            coords1=self.coords_per_dim[1],
        )
        logger.success(f"Adaptive policy saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path | str) -> "AdaptiveCudaPI2D":
        filepath = Path(filepath).with_suffix(".npz")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.value_function = data["value_function"]
        instance.policy         = data["policy"]
        instance.bounds_low     = data["bounds_low"]
        instance.bounds_high    = data["bounds_high"]
        instance.grid_shape     = data["grid_shape"]
        instance.strides        = data["strides"]
        instance.corner_bits    = data["corner_bits"]
        instance.action_space   = data["action_space"]
        instance.states_space   = data["states_space"]
        instance.n_actions      = len(instance.action_space)
        instance.n_states       = len(instance.states_space)
        instance.config         = CudaPIConfig()
        instance.adaptive_config = AdaptivePIConfig()
        instance.coords_per_dim = [data["coords0"], data["coords1"]]
        instance.dim_names      = ["dim0", "dim1"]

        logger.success(f"Adaptive policy loaded from {filepath.resolve()}")
        return instance


# ══════════════════════════════════════════════════════════════════════════════
# 4D adaptive mesh (e.g. CartPole: [x, x_dot, theta, theta_dot])
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveCudaPI4D(CudaPolicyIteration4D):
    """
    Hessian-based adaptive mesh Policy Iteration for 4D continuous state spaces.

    Same adaptive logic as AdaptiveCudaPI2D but extended to 4 dimensions:
    - 16-corner hypercube barycentric with binary-search per dim
    - Hessian estimated via diagonal finite differences (sufficient for
      per-dimension equidistribution)
    - Equidistribution: marginal curvature = max over the other 3 dims

    Subclass interface (same as CudaPolicyIteration4D):
      - Override _dynamics_cuda_src()  → step_dynamics __device__ function
      - Override _terminal_fn(states)  → (bool_mask, terminal_value)
    """

    def __init__(
        self,
        bins_space: dict,
        action_space: np.ndarray,
        config: CudaPIConfig | None = None,
        adaptive_config: AdaptivePIConfig | None = None,
    ) -> None:
        self.adaptive_config = adaptive_config or AdaptivePIConfig()
        keys = list(bins_space.keys())
        assert len(keys) == 4
        self.coords_per_dim = [bins_space[k].copy().astype(np.float32) for k in keys]
        self.dim_names = keys
        super().__init__(bins_space, action_space, config)

    # ── Grid metadata (override) ───────────────────────────────────────────────

    def _precompute_grid_metadata(self) -> None:
        self.bounds_low  = np.array([c[0]  for c in self.coords_per_dim], dtype=np.float32)
        self.bounds_high = np.array([c[-1] for c in self.coords_per_dim], dtype=np.float32)
        self.grid_shape  = np.array([len(c) for c in self.coords_per_dim], dtype=np.int32)
        n = self.grid_shape
        self.strides = np.array(
            [n[1]*n[2]*n[3], n[2]*n[3], n[3], 1], dtype=np.int32
        )
        from itertools import product as iproduct
        self.corner_bits = np.array(list(iproduct([0, 1], repeat=4)), dtype=np.int32)
        logger.info(
            f"Adaptive 4D grid: shape={self.grid_shape.tolist()}, "
            f"states={self.n_states:,}, actions={self.n_actions}"
        )

    # ── GPU allocation (override) ──────────────────────────────────────────────

    def _allocate_tensors_and_compile(self) -> None:
        super()._allocate_tensors_and_compile()
        self.d_coords0 = cp.asarray(self.coords_per_dim[0], dtype=cp.float32)
        self.d_coords1 = cp.asarray(self.coords_per_dim[1], dtype=cp.float32)
        self.d_coords2 = cp.asarray(self.coords_per_dim[2], dtype=cp.float32)
        self.d_coords3 = cp.asarray(self.coords_per_dim[3], dtype=cp.float32)

    # ── CUDA module (override) ─────────────────────────────────────────────────

    def _compile_cuda_module(self) -> None:
        adaptive_kernels = r'''
        extern "C" {

        __device__ int find_cell_4d(const float* coords, int n, float val) {
            if (val <= coords[0])   return 0;
            if (val >= coords[n-1]) return n - 2;
            int lo = 0, hi = n - 2;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (coords[mid] <= val) lo = mid;
                else                    hi = mid - 1;
            }
            return lo;
        }

        __device__ void get_barycentric_4d_adaptive(
            float s0, float s1, float s2, float s3,
            const float* c0, const float* c1, const float* c2, const float* c3,
            int n0, int n1, int n2, int n3, const int* strides,
            int* idxs, float* wgts
        ) {
            int i0 = find_cell_4d(c0, n0, s0);
            int i1 = find_cell_4d(c1, n1, s1);
            int i2 = find_cell_4d(c2, n2, s2);
            int i3 = find_cell_4d(c3, n3, s3);

            float t0 = fmaxf(0.f, fminf(1.f, (s0-c0[i0])/(c0[i0+1]-c0[i0])));
            float t1 = fmaxf(0.f, fminf(1.f, (s1-c1[i1])/(c1[i1+1]-c1[i1])));
            float t2 = fmaxf(0.f, fminf(1.f, (s2-c2[i2])/(c2[i2+1]-c2[i2])));
            float t3 = fmaxf(0.f, fminf(1.f, (s3-c3[i3])/(c3[i3+1]-c3[i3])));

            for (int c = 0; c < 16; ++c) {
                int b0 = (c >> 0) & 1, b1 = (c >> 1) & 1;
                int b2 = (c >> 2) & 1, b3 = (c >> 3) & 1;
                idxs[c] = (i0+b0)*strides[0] + (i1+b1)*strides[1]
                         +(i2+b2)*strides[2] + (i3+b3)*strides[3];
                wgts[c] = (b0?t0:1-t0)*(b1?t1:1-t1)*(b2?t2:1-t2)*(b3?t3:1-t3);
            }
        }

        // Diagonal-only Hessian norm (sufficient for per-dim equidistribution).
        // Uses non-uniform finite differences on each axis independently.
        __global__ void compute_hessian_norm_4d(
            const float* V,
            const float* c0, const float* c1, const float* c2, const float* c3,
            float* hess_norm,
            int n0, int n1, int n2, int n3
        ) {
            int s = blockIdx.x * blockDim.x + threadIdx.x;
            if (s >= n0*n1*n2*n3) return;

            int i3 =  s % n3;
            int i2 = (s / n3) % n2;
            int i1 = (s / (n3*n2)) % n1;
            int i0 =  s / (n3*n2*n1);

            if (i0==0||i0==n0-1||i1==0||i1==n1-1||
                i2==0||i2==n2-1||i3==0||i3==n3-1) {
                hess_norm[s] = 0.0f; return;
            }

            int st0=n1*n2*n3, st1=n2*n3, st2=n3, st3=1;
            float Vc = V[s];

            float hl0=c0[i0]-c0[i0-1], hr0=c0[i0+1]-c0[i0];
            float hl1=c1[i1]-c1[i1-1], hr1=c1[i1+1]-c1[i1];
            float hl2=c2[i2]-c2[i2-1], hr2=c2[i2+1]-c2[i2];
            float hl3=c3[i3]-c3[i3-1], hr3=c3[i3+1]-c3[i3];

            float d0=2*(V[s+st0]/hr0-Vc*(hl0+hr0)/(hl0*hr0)+V[s-st0]/hl0)/(hl0+hr0);
            float d1=2*(V[s+st1]/hr1-Vc*(hl1+hr1)/(hl1*hr1)+V[s-st1]/hl1)/(hl1+hr1);
            float d2=2*(V[s+st2]/hr2-Vc*(hl2+hr2)/(hl2*hr2)+V[s-st2]/hl2)/(hl2+hr2);
            float d3=2*(V[s+st3]/hr3-Vc*(hl3+hr3)/(hl3*hr3)+V[s-st3]/hl3)/(hl3+hr3);

            hess_norm[s] = sqrtf(d0*d0 + d1*d1 + d2*d2 + d3*d3);
        }

        __global__ void policy_eval_kernel_4d_adaptive(
            const float* states, const float* actions, const int* policy,
            const float* V, float* new_V, const bool* is_term,
            const float* c0, const float* c1, const float* c2, const float* c3,
            const int* g_shape, const int* strides,
            int n_states, float gamma_discount
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) { new_V[s_idx] = V[s_idx]; return; }

            float s0=states[s_idx*4+0], s1=states[s_idx*4+1];
            float s2=states[s_idx*4+2], s3=states[s_idx*4+3];
            float action = actions[policy[s_idx]];

            float ns0,ns1,ns2,ns3,reward; bool terminated;
            step_dynamics(s0,s1,s2,s3,action,&ns0,&ns1,&ns2,&ns3,&reward,&terminated);

            float expected_v = 0.0f;
            if (!terminated) {
                int idxs[16]; float wgts[16];
                get_barycentric_4d_adaptive(ns0,ns1,ns2,ns3,c0,c1,c2,c3,
                    g_shape[0],g_shape[1],g_shape[2],g_shape[3],strides,idxs,wgts);
                for (int i = 0; i < 16; ++i)
                    expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
            }
            new_V[s_idx] = reward + gamma_discount * expected_v;
        }

        __global__ void policy_improve_kernel_4d_adaptive(
            const float* states, const float* actions, int* policy,
            const float* V, const bool* is_term,
            const float* c0, const float* c1, const float* c2, const float* c3,
            const int* g_shape, const int* strides,
            int n_states, int n_actions, float gamma_discount
        ) {
            int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (s_idx >= n_states) return;
            if (is_term[s_idx]) return;

            float s0=states[s_idx*4+0], s1=states[s_idx*4+1];
            float s2=states[s_idx*4+2], s3=states[s_idx*4+3];

            float max_q = -1.0e30f; int best_a = 0;

            for (int a = 0; a < n_actions; ++a) {
                float action = actions[a];
                float ns0,ns1,ns2,ns3,reward; bool terminated;
                step_dynamics(s0,s1,s2,s3,action,&ns0,&ns1,&ns2,&ns3,&reward,&terminated);

                float expected_v = 0.0f;
                if (!terminated) {
                    int idxs[16]; float wgts[16];
                    get_barycentric_4d_adaptive(ns0,ns1,ns2,ns3,c0,c1,c2,c3,
                        g_shape[0],g_shape[1],g_shape[2],g_shape[3],strides,idxs,wgts);
                    for (int i = 0; i < 16; ++i)
                        expected_v = fmaf(wgts[i], V[idxs[i]], expected_v);
                }
                float q = reward + gamma_discount * expected_v;
                if (q > max_q) { max_q = q; best_a = a; }
            }
            policy[s_idx] = best_a;
        }

        } // extern "C"
        '''

        cuda_source = self._dynamics_cuda_src() + adaptive_kernels
        module = cp.RawModule(code=cuda_source)

        self.hessian_kernel = module.get_function("compute_hessian_norm_4d")
        self.eval_kernel    = module.get_function("policy_eval_kernel_4d_adaptive")
        self.improve_kernel = module.get_function("policy_improve_kernel_4d_adaptive")

        self.threads_per_block = 256
        self.blocks_per_grid = (
            self.n_states + self.threads_per_block - 1
        ) // self.threads_per_block

    # ── Policy iteration overrides ─────────────────────────────────────────────

    def policy_evaluation(self) -> float:
        delta = float("inf")
        SYNC_INTERVAL = 25
        for i in range(self.config.max_eval_iter):
            self.eval_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (
                    self.d_states, self.d_actions, self.d_policy,
                    self.d_value_function, self.d_new_value_function,
                    self.d_terminal_mask,
                    self.d_coords0, self.d_coords1, self.d_coords2, self.d_coords3,
                    self.d_grid_shape, self.d_strides,
                    np.int32(self.n_states), np.float32(self.config.gamma),
                )
            )
            d_delta = self._max_abs_diff(self.d_new_value_function, self.d_value_function)
            self.d_value_function, self.d_new_value_function = (
                self.d_new_value_function, self.d_value_function
            )
            if i % SYNC_INTERVAL == 0 or i == self.config.max_eval_iter - 1:
                delta = float(d_delta.get())
                if i % self.config.log_interval == 0:
                    logger.debug(f"  Eval iter {i:5d} | delta = {delta:.4e}")
                if delta < self.config.theta:
                    logger.success(f"  Eval converged at iter {i} | delta = {delta:.2e}")
                    return delta
        logger.warning(f"  Eval hit max_eval_iter={self.config.max_eval_iter} | delta = {delta:.2e}")
        return delta

    def policy_improvement(self) -> bool:
        old_policy = self.d_policy.copy()
        self.improve_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_states, self.d_actions, self.d_policy,
                self.d_value_function, self.d_terminal_mask,
                self.d_coords0, self.d_coords1, self.d_coords2, self.d_coords3,
                self.d_grid_shape, self.d_strides,
                np.int32(self.n_states), np.int32(self.n_actions),
                np.float32(self.config.gamma),
            )
        )
        return bool(cp.all(self.d_policy == old_policy))

    # ── Inner PI loop ──────────────────────────────────────────────────────────

    def _run_pi_loop(self) -> None:
        for n in range(self.config.max_pi_iter):
            logger.info(f"-- PI Iteration {n+1}/{self.config.max_pi_iter} --")
            self.policy_evaluation()
            if self.policy_improvement():
                logger.success(f"PI converged at iteration {n+1}.")
                return
        logger.warning(f"PI hit max_pi_iter={self.config.max_pi_iter}.")

    # ── Adaptive mesh steps ────────────────────────────────────────────────────

    def compute_hessian_norms(self) -> np.ndarray:
        d_hess = cp.zeros(self.n_states, dtype=cp.float32)
        n0,n1,n2,n3 = [int(x) for x in self.grid_shape]
        self.hessian_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                self.d_value_function,
                self.d_coords0, self.d_coords1, self.d_coords2, self.d_coords3,
                d_hess,
                np.int32(n0), np.int32(n1), np.int32(n2), np.int32(n3),
            )
        )
        result = d_hess.get()
        d_hess = None
        cp.get_default_memory_pool().free_all_blocks()
        return result

    def equidistribute_nodes(
        self, hess_norm: np.ndarray, n_new: int
    ) -> list[np.ndarray]:
        cfg = self.adaptive_config
        hess_4d = hess_norm.reshape(self.grid_shape)
        new_coords = []

        for d in range(4):
            other = tuple(i for i in range(4) if i != d)
            curvature_1d = hess_4d.max(axis=other)

            # Normalize to [0, 1] — same rationale as 2D version.
            h_max = curvature_1d.max()
            if h_max > 0:
                curvature_1d = curvature_1d / h_max

            density = np.sqrt((curvature_1d + cfg.nu) / cfg.epsilon)
            density = np.maximum(density, 1e-8)

            cumulative = np.cumsum(density)
            cumulative /= cumulative[-1]

            old_coords = self.coords_per_dim[d]
            targets = np.linspace(0.0, 1.0, n_new)
            new_c = np.interp(targets, cumulative, old_coords).astype(np.float32)
            new_c[0]  = old_coords[0]
            new_c[-1] = old_coords[-1]

            # Enforce strict monotonicity using np.nextafter (float32-safe).
            hi = np.float32(old_coords[-1] + 1.0)
            for k in range(1, len(new_c) - 1):
                if new_c[k] <= new_c[k - 1]:
                    new_c[k] = np.nextafter(new_c[k - 1], hi)
            new_c[-1] = old_coords[-1]

            new_coords.append(new_c)

        return new_coords

    def transfer_value_function(
        self, new_coords: list[np.ndarray]
    ) -> np.ndarray:
        V_4d = self.d_value_function.get().reshape(self.grid_shape)
        interp = RegularGridInterpolator(
            tuple(self.coords_per_dim),
            V_4d,
            method='linear',
            bounds_error=False,
            fill_value=None,
        )
        g = np.meshgrid(*new_coords, indexing='ij')
        pts = np.stack([gi.ravel() for gi in g], axis=-1)
        return interp(pts).astype(np.float32)

    def rebuild_grid(
        self, new_coords: list[np.ndarray], V_init: np.ndarray
    ) -> None:
        _gpu_attrs = [
            "d_states", "d_actions", "d_bounds_low", "d_bounds_high",
            "d_grid_shape", "d_strides", "d_terminal_mask",
            "d_value_function", "d_new_value_function", "d_policy",
            "d_coords0", "d_coords1", "d_coords2", "d_coords3",
        ]
        for attr in _gpu_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        cp.get_default_memory_pool().free_all_blocks()

        self.coords_per_dim = new_coords
        g = np.meshgrid(*new_coords, indexing='ij')
        self.states_space = np.column_stack(
            [gi.ravel() for gi in g]
        ).astype(np.float32)
        self.n_states = len(self.states_space)

        self._precompute_grid_metadata()
        self._allocate_tensors_and_compile()

        self.d_value_function[:]     = cp.asarray(V_init, dtype=cp.float32)
        self.d_new_value_function[:] = self.d_value_function[:]

        terminal_mask, terminal_value = self._terminal_fn(self.states_space)
        if np.any(terminal_mask):
            d_term = cp.asarray(terminal_mask, dtype=cp.bool_)
            self.d_value_function[d_term]     = float(terminal_value)
            self.d_new_value_function[d_term] = float(terminal_value)

        logger.info(
            f"Grid rebuilt -> shape={self.grid_shape.tolist()}, "
            f"{self.n_states:,} states"
        )

    # ── Main adaptive loop ─────────────────────────────────────────────────────

    def run_adaptive(self) -> None:
        cfg = self.adaptive_config

        logger.info("== Phase 1: Coarse uniform solve ==")
        self._run_pi_loop()

        for k in range(cfg.n_refine):
            logger.info(f"== Refinement {k+1}/{cfg.n_refine} ==")

            logger.info("  [2] Computing Hessian norms on GPU...")
            hess = self.compute_hessian_norms()
            logger.info(
                f"      ||d2V|| min={hess.min():.3f}  "
                f"max={hess.max():.3f}  mean={hess.mean():.3f}"
            )

            logger.info(f"  [3] Equidistributing {cfg.n_nodes_refined} nodes/dim...")
            new_coords = self.equidistribute_nodes(hess, cfg.n_nodes_refined)

            logger.info("  [4] Transferring V to new grid...")
            V_new = self.transfer_value_function(new_coords)

            logger.info("  [5] Rebuilding grid and re-solving...")
            self.rebuild_grid(new_coords, V_new)
            self._run_pi_loop()

        self._pull_tensors_from_gpu()
        logger.success("Adaptive 4D mesh policy iteration complete.")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: Path | str) -> None:
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
            coords0=self.coords_per_dim[0],
            coords1=self.coords_per_dim[1],
            coords2=self.coords_per_dim[2],
            coords3=self.coords_per_dim[3],
        )
        logger.success(f"Adaptive 4D policy saved to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path | str) -> "AdaptiveCudaPI4D":
        filepath = Path(filepath).with_suffix(".npz")
        data = np.load(filepath)

        instance = cls.__new__(cls)
        instance.value_function  = data["value_function"]
        instance.policy          = data["policy"]
        instance.bounds_low      = data["bounds_low"]
        instance.bounds_high     = data["bounds_high"]
        instance.grid_shape      = data["grid_shape"]
        instance.strides         = data["strides"]
        instance.corner_bits     = data["corner_bits"]
        instance.action_space    = data["action_space"]
        instance.states_space    = data["states_space"]
        instance.n_actions       = len(instance.action_space)
        instance.n_states        = len(instance.states_space)
        instance.config          = CudaPIConfig()
        instance.adaptive_config = AdaptivePIConfig()
        instance.coords_per_dim  = [
            data["coords0"], data["coords1"],
            data["coords2"], data["coords3"],
        ]
        instance.dim_names = ["dim0", "dim1", "dim2", "dim3"]

        logger.success(f"Adaptive 4D policy loaded from {filepath.resolve()}")
        return instance
