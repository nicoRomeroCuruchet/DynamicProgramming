"""
Score function for autoresearch — DO NOT MODIFY.

This file is the *external* metric used to score each autoresearch trial.
The agent under autoresearch can edit `double_cartpole_swingup_cuda.py`
freely, but this file is the source of truth for which trial wins.

Run as a script after `evaluate` has dumped `results/last_trajectory.npz`:

    python runners/eval_metric.py

Outputs a single line: SCORE=<float>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


TRAJ_PATH = Path("results") / "last_trajectory.npz"


def score(traj: np.ndarray) -> dict[str, float]:
    """Return a dict of metrics. Higher SCORE is better.

    Parameters
    ----------
    traj : (N, 6) float32 — concatenated steady-state windows from all
           evaluation episodes. Columns are [x, x_dot, th1, w1, th2, w2].

    Metric design
    -------------
    - frac_deep_upright: fraction of timesteps where BOTH cos(th_i) > 0.7
      (~45 degrees from vertical). Robust to reward hacking — the only
      way to maximize this is to actually keep both poles upright.
    - avg_gate: mean(max(0, c1) * max(0, c2)). Smooth tiebreaker so
      partial progress is rewarded.
    - cart_safety: penalty if the cart spends time near the bounds.

    Final SCORE = frac_deep_upright + 0.1 * avg_gate - 0.05 * cart_unsafe.
    """
    cos1 = np.cos(traj[:, 2])
    cos2 = np.cos(traj[:, 4])

    deep_upright       = (cos1 > 0.7) & (cos2 > 0.7)
    frac_deep_upright  = float(deep_upright.mean())
    avg_gate           = float(
        (np.maximum(0.0, cos1) * np.maximum(0.0, cos2)).mean()
    )

    x = traj[:, 0]
    cart_unsafe = float((np.abs(x) > 2.0).mean())

    score_value = frac_deep_upright + 0.1 * avg_gate - 0.05 * cart_unsafe

    return {
        "score":             score_value,
        "frac_deep_upright": frac_deep_upright,
        "avg_gate":          avg_gate,
        "cart_unsafe":       cart_unsafe,
        "mean_cos1":         float(cos1.mean()),
        "mean_cos2":         float(cos2.mean()),
        "n_steps":           int(traj.shape[0]),
    }


def main() -> int:
    if not TRAJ_PATH.exists():
        print(f"ERROR: trajectory file not found: {TRAJ_PATH}")
        print("       Run `evaluate` first.")
        return 1

    data = np.load(TRAJ_PATH)
    traj = data["traj"]
    metrics = score(traj)

    print(f"SCORE={metrics['score']:.6f}")
    print(f"  frac_deep_upright = {metrics['frac_deep_upright']:.4f}")
    print(f"  avg_gate          = {metrics['avg_gate']:.4f}")
    print(f"  cart_unsafe       = {metrics['cart_unsafe']:.4f}")
    print(f"  mean_cos1         = {metrics['mean_cos1']:+.4f}")
    print(f"  mean_cos2         = {metrics['mean_cos2']:+.4f}")
    print(f"  n_steps           = {metrics['n_steps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
