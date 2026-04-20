"""
runners/mountain_car_benchmark.py — Value function accuracy benchmark.

Core question: for a fixed state budget N², which approach approximates the
true value function V* more accurately — uniform N×N grid or adaptive N×N?

Method:
  1. Compute V_ref on a fine reference grid (REF_BINS × REF_BINS).
  2. Train each configuration (uniform / adaptive) at various state budgets.
  3. Interpolate each V_h onto the reference grid.
  4. Report L2 and L∞ errors relative to V_ref (non-terminal states only).

Configurations:
  Uniform  : 30, 50, 75, 100, 150, 200 nodes/dim
  Adaptive : coarse COARSE_BINS → refined to 50, 75, 100, 150, 200 nodes/dim

Usage:
    python3 runners/mountain_car_benchmark.py
    python3 runners/mountain_car_benchmark.py --coarse 40 --ref-bins 600
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cupy as cp
from loguru import logger

from src.cuda_policy_iteration import CudaPolicyIteration2D, CudaPIConfig
from src.adaptive_cuda_pi import AdaptiveCudaPI2D, AdaptivePIConfig

logger.disable("src")

# ── Reference grid ─────────────────────────────────────────────────────────────
REF_BINS = 500   # fine reference grid (REF_BINS² states ≈ 250k)

# ── Shared dynamics ────────────────────────────────────────────────────────────

_DYNAMICS = r'''
__device__ void step_dynamics(
    float pos, float vel, float force,
    float* next_pos, float* next_vel,
    float* reward, bool* terminated
) {
    vel += force * 0.001f - 0.0025f * cosf(3.0f * pos);
    vel  = fmaxf(-0.07f, fminf(0.07f, vel));
    pos += vel;
    pos  = fmaxf(-1.2f,  fminf(0.6f,  pos));
    if (pos <= -1.2f) vel = 0.0f;
    *next_pos   = pos;
    *next_vel   = vel;
    *terminated = (pos >= 0.5f) && (vel >= 0.0f);
    *reward     = -1.0f;
}
'''

ACTION_SPACE = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


def _terminal_fn(states):
    return (states[:, 0] >= 0.5) & (states[:, 1] >= 0.0), 0.0


class _UniformMC(CudaPolicyIteration2D):
    def _dynamics_cuda_src(self): return _DYNAMICS
    def _terminal_fn(self, s):    return _terminal_fn(s)


class _AdaptiveMC(AdaptiveCudaPI2D):
    def _dynamics_cuda_src(self): return _DYNAMICS
    def _terminal_fn(self, s):    return _terminal_fn(s)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bins(n: int) -> dict:
    return {
        "position": np.linspace(-1.2,  0.6,  n, dtype=np.float32),
        "velocity": np.linspace(-0.07, 0.07, n, dtype=np.float32),
    }


def _model_memory_mb(n_states: int, n_actions: int) -> float:
    """Analytical GPU memory for the model tensors (float32/int32/bool)."""
    bytes_ = (
        n_states * 2 * 4   # states (float32)
      + n_states * 4       # value_function
      + n_states * 4       # new_value_function
      + n_states * 4       # policy (int32)
      + n_states * 1       # terminal_mask (bool)
      + n_actions * 4      # action_space
    )
    return bytes_ / 1024**2


def _free_gpu():
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


# ── Result ─────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    label:      str
    kind:       str       # "uniform" | "adaptive" | "reference"
    n_nodes:    int       # nodes per dim (final grid)
    n_states:   int
    t_train:    float
    mem_mb:     float     # analytical model memory
    err_l2:     float = 0.0   # relative L2: ‖V_h - V_ref‖₂ / ‖V_ref‖₂
    err_linf:   float = 0.0   # L∞: max|V_h - V_ref|
    err_l1:     float = 0.0   # mean absolute error
    coords:     list  = field(default_factory=list)
    V:          Optional[np.ndarray] = None


# ── Training ───────────────────────────────────────────────────────────────────

def train_reference(ref_bins: int, pi_cfg: CudaPIConfig) -> RunResult:
    """High-resolution reference solve."""
    _free_gpu()
    t0 = time.perf_counter()
    pi = _UniformMC(_make_bins(ref_bins), ACTION_SPACE, pi_cfg)
    pi.run()
    t = time.perf_counter() - t0

    c0 = np.unique(pi.states_space[:, 0])
    c1 = np.unique(pi.states_space[:, 1])
    V  = pi.value_function.reshape(pi.grid_shape)

    print(f"  [ref] {ref_bins}×{ref_bins} = {pi.n_states:,} states | {t:.1f}s")
    return RunResult(
        label=f"Reference {ref_bins}×{ref_bins}", kind="reference",
        n_nodes=ref_bins, n_states=pi.n_states,
        t_train=t, mem_mb=_model_memory_mb(pi.n_states, len(ACTION_SPACE)),
        coords=[c0, c1], V=V,
    )


def train_uniform(n: int, pi_cfg: CudaPIConfig) -> RunResult:
    _free_gpu()
    t0 = time.perf_counter()
    pi = _UniformMC(_make_bins(n), ACTION_SPACE, pi_cfg)
    pi.run()
    t = time.perf_counter() - t0

    c0 = np.unique(pi.states_space[:, 0])
    c1 = np.unique(pi.states_space[:, 1])
    V  = pi.value_function.reshape(pi.grid_shape)

    return RunResult(
        label=f"Uniform {n}×{n}", kind="uniform",
        n_nodes=n, n_states=pi.n_states,
        t_train=t, mem_mb=_model_memory_mb(pi.n_states, len(ACTION_SPACE)),
        coords=[c0, c1], V=V,
    )


def train_adaptive(n: int, coarse: int,
                   pi_cfg: CudaPIConfig, adp_cfg: AdaptivePIConfig) -> RunResult:
    _free_gpu()
    t0 = time.perf_counter()
    pi = _AdaptiveMC(_make_bins(coarse), ACTION_SPACE, pi_cfg, adp_cfg)
    pi.run_adaptive()
    t = time.perf_counter() - t0

    c0, c1 = pi.coords_per_dim
    V = pi.value_function.reshape(pi.grid_shape)

    return RunResult(
        label=f"Adaptive {coarse}→{n}", kind="adaptive",
        n_nodes=n, n_states=pi.n_states,
        t_train=t, mem_mb=_model_memory_mb(pi.n_states, len(ACTION_SPACE)),
        coords=[c0, c1], V=V,
    )


# ── Error computation ──────────────────────────────────────────────────────────

def compute_errors(r: RunResult, ref: RunResult) -> None:
    """
    Interpolate r.V onto the reference grid, then compute:
      - L2 relative error
      - L∞ absolute error
      - L1 mean absolute error
    Only non-terminal states are included (terminal V_ref ≈ 0 trivially).
    """
    interp = RegularGridInterpolator(
        (r.coords[0], r.coords[1]), r.V,
        method="linear", bounds_error=False, fill_value=None,
    )

    ref_c0, ref_c1 = ref.coords
    g0, g1 = np.meshgrid(ref_c0, ref_c1, indexing="ij")
    pts = np.stack([g0.ravel(), g1.ravel()], axis=-1)

    V_h   = interp(pts).reshape(ref.V.shape)
    V_ref = ref.V

    # Mask out terminal states (V_ref == 0 at terminal)
    nonterminal = V_ref < -0.1    # V is always ≤ -1 at non-terminal states

    diff = V_h[nonterminal] - V_ref[nonterminal]
    r.err_l2   = float(np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean(V_ref[nonterminal]**2)))
    r.err_linf = float(np.max(np.abs(diff)))
    r.err_l1   = float(np.mean(np.abs(diff)))


# ── Report ─────────────────────────────────────────────────────────────────────

def print_report(results: list[RunResult]) -> None:
    sep = "─" * 76
    hdr = f"{'Label':<22} {'States':>8} {'Time(s)':>8} {'Mem(MB)':>8} " \
          f"{'L2 rel':>9} {'L∞ abs':>9} {'L1 abs':>9}"

    print(f"\n{'═'*76}")
    print("  Mountain Car — Value Function Error  (‖V_h - V_ref‖)")
    print(f"{'═'*76}")
    print(f"  {hdr}")
    print(f"  {sep}")

    for r in results:
        if r.kind == "reference":
            continue
        marker = "▶" if r.kind == "adaptive" else " "
        print(
            f"  {marker} {r.label:<20} {r.n_states:>8,} {r.t_train:>8.2f} "
            f"{r.mem_mb:>8.3f} "
            f"{r.err_l2:>9.4f} {r.err_linf:>9.2f} {r.err_l1:>9.2f}"
        )
    print(f"{'═'*76}\n")


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_benchmark(
    results: list[RunResult],
    ref: RunResult,
    save_path: Path = Path("results/mountain_car_benchmark.png"),
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    uniform  = [r for r in results if r.kind == "uniform"]
    adaptive = [r for r in results if r.kind == "adaptive"]

    u_states = [r.n_states for r in uniform]
    a_states = [r.n_states for r in adaptive]

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # ── A) L2 relative error vs states ──
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(u_states, [r.err_l2 * 100 for r in uniform],
            "o-", color="steelblue", label="Uniform", linewidth=2, markersize=6)
    ax.plot(a_states, [r.err_l2 * 100 for r in adaptive],
            "s--", color="darkorange", label="Adaptive", linewidth=2, markersize=6)
    ax.set_xlabel("States (N²)"); ax.set_ylabel("L2 relative error (%)")
    ax.set_title("A) L2 error  ‖V_h - V_ref‖₂ / ‖V_ref‖₂")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # ── B) L∞ error vs states ──
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(u_states, [r.err_linf for r in uniform],
            "o-", color="steelblue", label="Uniform", linewidth=2, markersize=6)
    ax.plot(a_states, [r.err_linf for r in adaptive],
            "s--", color="darkorange", label="Adaptive", linewidth=2, markersize=6)
    ax.set_xlabel("States (N²)"); ax.set_ylabel("L∞ error  max|V_h - V_ref|")
    ax.set_title("B) L∞ error  (worst-case pointwise)")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # ── C) Training time vs states ──
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(u_states, [r.t_train for r in uniform],
            "o-", color="steelblue", label="Uniform", linewidth=2, markersize=6)
    ax.plot(a_states, [r.t_train for r in adaptive],
            "s--", color="darkorange", label="Adaptive", linewidth=2, markersize=6)
    ax.set_xlabel("States (N²)"); ax.set_ylabel("Training time (s)")
    ax.set_title("C) Training time")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xscale("log")

    # ── D) Error vs time (Pareto frontier) ──
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter([r.t_train for r in uniform],
               [r.err_l2 * 100 for r in uniform],
               color="steelblue", label="Uniform", s=80,
               zorder=3)
    ax.scatter([r.t_train for r in adaptive],
               [r.err_l2 * 100 for r in adaptive],
               color="darkorange", marker="s", label="Adaptive", s=80,
               zorder=3)
    for r in uniform + adaptive:
        ax.annotate(f"{r.n_states//1000}k",
                    (r.t_train, r.err_l2 * 100),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Training time (s)"); ax.set_ylabel("L2 error (%)")
    ax.set_title("D) Error vs time  (lower-left = better)")
    ax.legend(); ax.grid(alpha=0.3)

    # ── E) V_ref heatmap ──
    ax = fig.add_subplot(gs[1, 1])
    c0, c1 = ref.coords
    im = ax.pcolormesh(c0, c1, ref.V.T, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="V")
    ax.set_xlabel("Position"); ax.set_ylabel("Velocity")
    ax.set_title(f"E) Reference V  ({ref.n_nodes}×{ref.n_nodes}, {ref.n_states:,} states)")

    # ── F) Pointwise error heatmap — smallest uniform vs adaptive at same budget ──
    # Find the adaptive result closest in n_states to smallest uniform
    r_u = uniform[0]
    r_a = min(adaptive, key=lambda r: abs(r.n_states - r_u.n_states))

    # Compute pointwise error on reference grid for both
    def _pointwise_err(r):
        interp = RegularGridInterpolator(
            (r.coords[0], r.coords[1]), r.V,
            method="linear", bounds_error=False, fill_value=None,
        )
        g0, g1 = np.meshgrid(ref.coords[0], ref.coords[1], indexing="ij")
        pts = np.stack([g0.ravel(), g1.ravel()], axis=-1)
        return np.abs(interp(pts).reshape(ref.V.shape) - ref.V)

    err_u = _pointwise_err(r_u)
    err_a = _pointwise_err(r_a)
    err_max = max(err_u.max(), err_a.max())

    ax = fig.add_subplot(gs[1, 2])
    # Show difference: positive = adaptive is better, negative = uniform is better
    diff_err = err_u - err_a
    lim = np.percentile(np.abs(diff_err), 98)
    im2 = ax.pcolormesh(ref.coords[0], ref.coords[1], diff_err.T,
                        cmap="RdBu", shading="auto", vmin=-lim, vmax=lim)
    fig.colorbar(im2, ax=ax, label="|err_uniform| - |err_adaptive|")
    ax.set_xlabel("Position"); ax.set_ylabel("Velocity")
    ax.set_title(
        f"F) Error difference  (blue = adaptive wins)\n"
        f"{r_u.label} vs {r_a.label}  (~{r_u.n_states:,} states each)"
    )

    plt.suptitle(
        "Mountain Car — Adaptive vs Uniform: Value Function Approximation Error\n"
        f"Reference: {ref.n_nodes}×{ref.n_nodes} uniform  |  "
        f"Error on non-terminal states only",
        fontsize=11, fontweight="bold",
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Benchmark plot saved to {save_path.resolve()}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mountain Car — value function error benchmark"
    )
    parser.add_argument("--coarse",    type=int, default=40,
                        help="Coarse bins for adaptive warm-start (default: 40)")
    parser.add_argument("--n-refine",  type=int, default=2,
                        help="Adaptive refinement cycles (default: 2)")
    parser.add_argument("--ref-bins",  type=int, default=REF_BINS,
                        help=f"Reference grid size (default: {REF_BINS})")
    parser.add_argument("--save-path", type=Path,
                        default=Path("results/mountain_car_benchmark.png"))
    args = parser.parse_args()

    pi_cfg = CudaPIConfig(
        gamma=0.99, theta=1e-4,
        max_eval_iter=5_000, max_pi_iter=50,
        log_interval=5_000,
    )

    uniform_sizes  = [30, 50, 75, 100, 150, 200]
    adaptive_sizes = [50, 75, 100, 150, 200]

    # ── Reference ──
    print(f"\nComputing reference V on {args.ref_bins}×{args.ref_bins} grid...")
    ref_pi_cfg = CudaPIConfig(
        gamma=0.99, theta=1e-5,
        max_eval_iter=10_000, max_pi_iter=100,
        log_interval=10_000,
    )
    ref = train_reference(args.ref_bins, ref_pi_cfg)

    results: list[RunResult] = []

    # ── Uniform ──
    print(f"\nTraining uniform grids: {uniform_sizes}")
    for n in uniform_sizes:
        r = train_uniform(n, pi_cfg)
        compute_errors(r, ref)
        results.append(r)
        print(f"  {r.label:<22} {r.n_states:>7,} states | "
              f"{r.t_train:.2f}s | L2={r.err_l2*100:.2f}%  L∞={r.err_linf:.2f}")

    # ── Adaptive ──
    print(f"\nTraining adaptive grids: coarse={args.coarse} → {adaptive_sizes}")
    for n in adaptive_sizes:
        adp_cfg = AdaptivePIConfig(
            epsilon=0.01, nu=0.1,
            n_refine=args.n_refine, n_nodes_refined=n,
        )
        r = train_adaptive(n, args.coarse, pi_cfg, adp_cfg)
        compute_errors(r, ref)
        results.append(r)
        print(f"  {r.label:<22} {r.n_states:>7,} states | "
              f"{r.t_train:.2f}s | L2={r.err_l2*100:.2f}%  L∞={r.err_linf:.2f}")

    print_report(results)
    plot_benchmark(results, ref, args.save_path)
