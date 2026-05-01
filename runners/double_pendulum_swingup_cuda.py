"""
runners/double_pendulum_swingup_cuda.py
CUDA Policy Iteration for Double Pendulum Swing-Up (no cart).

This is the Pendulum-v1 analog with two links: a fixed pivot at the origin,
two coupled links of point-mass m1, m2 at lengths l1, l2, and a single
control torque applied at the BASE joint (link 1 to ground). Link 2 is
NOT directly actuated -- it only moves through the inertial coupling with
link 1, which makes the swing-up underactuated and non-trivial.

State  : [theta1   in [-pi, pi]   first  link angle from upright
          omega1   in [-15,  15]  first  link angular velocity
          theta2   in [-pi, pi]   second link angle from upright
          omega2   in [-15,  15]  second link angular velocity]

Action : torque applied at the base joint, discretized values in N*m.

Goal   : starting from hanging (theta1 = theta2 = pi), drive both links to
         upright (theta1 = theta2 = 0) and stabilize there.

Why this exists
---------------
The double-cartpole runner is 6D (cart + 2 angles + 3 velocities). At
bins=20 that is 64M states; one trial takes 30+ minutes. Removing the cart
drops to 4D -- bins=15 is 50K states, bins=25 is 390K states, both train
in seconds-to-minutes. Useful for:
  - rapid reward-shaping iteration (and as the autoresearch sandbox)
  - validating multi-GPU plumbing on a small problem first
  - studying underactuated swing-up in isolation from cart dynamics

Tuning notes (BINS_PER_DIM):
  bins=15 -> 50K states  -> ~5 sec on RTX 3090
  bins=20 -> 160K states -> ~30 sec
  bins=30 -> 810K states -> ~5 min
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPIConfig, CudaPolicyIteration4D


# Grid & action space
# --------------------------------------------------------------------------

BINS_PER_DIM = 15

BINS_SPACE = {
    "theta1":  np.linspace(-np.pi, np.pi, BINS_PER_DIM, dtype=np.float32),
    "th1_dot": np.linspace(-15.0,  15.0,  BINS_PER_DIM, dtype=np.float32),
    "theta2":  np.linspace(-np.pi, np.pi, BINS_PER_DIM, dtype=np.float32),
    "th2_dot": np.linspace(-15.0,  15.0,  BINS_PER_DIM, dtype=np.float32),
}

# Discrete torques at the base joint (N*m). The fine zero-crossing values
# (+/-0.5) help fine balancing once near upright; the wide ones (+/-3) are
# needed to pump enough energy through the coupling into link 2.
ACTION_SPACE = np.array(
    [-3.0, -1.5, -0.5, -0.15, -0.05, 0.0, 0.05, 0.15, 0.5, 1.5, 3.0],
    dtype=np.float32,
)


# CUDA subclass
# --------------------------------------------------------------------------

class DoublePendulumSwingUpCuda(CudaPolicyIteration4D):
    """Two-link pendulum swing-up, base-actuated."""

    def _dynamics_cuda_src(self) -> str:
        return r'''
        #define DP_G          9.8f
        #define DP_M1         0.1f
        #define DP_M2         0.1f
        #define DP_L1         0.5f
        #define DP_L2         0.5f
        #define DP_TAU        0.02f
        #define DP_PI         3.14159265358979323846f

        // E_target = (m1+m2)*g*l1 + m2*g*l2 = mechanical energy at upright rest.
        #define DP_E_TARGET   ((DP_M1 + DP_M2) * DP_G * DP_L1 + DP_M2 * DP_G * DP_L2)

        __device__ float dp_wrap(float x) {
            float r = fmodf(x + DP_PI, 2.0f * DP_PI);
            if (r < 0.0f) r += 2.0f * DP_PI;
            return r - DP_PI;
        }

        __device__ void step_dynamics(
            float th1, float w1, float th2, float w2, float u,
            float* nth1, float* nw1, float* nth2, float* nw2,
            float* reward, bool* terminated
        ) {
            float m12   = DP_M1 + DP_M2;
            float dth   = th1 - th2;
            float c12   = cosf(dth);
            float s12   = sinf(dth);

            // 2x2 mass matrix M:
            //   M11 = (m1+m2)*l1^2
            //   M12 = M21 = m2*l1*l2*cos(th1 - th2)
            //   M22 = m2*l2^2
            float M11 = m12 * DP_L1 * DP_L1;
            float M12 = DP_M2 * DP_L1 * DP_L2 * c12;
            float M22 = DP_M2 * DP_L2 * DP_L2;

            // RHS b: M * [w1_dot, w2_dot]^T = b
            //   b1 = tau1 + (m1+m2)*g*l1*sin(th1) - m2*l1*l2*sin(th1-th2)*w2^2
            //   b2 = tau2 + m2*g*l2*sin(th2)     + m2*l1*l2*sin(th1-th2)*w1^2
            // Underactuated: tau1 = u (base torque), tau2 = 0.
            float b1 = u
                     + m12 * DP_G * DP_L1 * sinf(th1)
                     - DP_M2 * DP_L1 * DP_L2 * s12 * w2 * w2;
            float b2 =        DP_M2 * DP_G * DP_L2 * sinf(th2)
                     + DP_M2 * DP_L1 * DP_L2 * s12 * w1 * w1;

            // Analytic 2x2 inverse. det = m2*l1^2*l2^2*(m1 + m2*sin^2(dth)) > 0
            float det  = M11 * M22 - M12 * M12;
            float w1d  = (M22 * b1 - M12 * b2) / det;
            float w2d  = (M11 * b2 - M12 * b1) / det;

            // Euler integration.
            *nth1 = dp_wrap(th1 + DP_TAU * w1);
            *nw1  = w1  + DP_TAU * w1d;
            *nth2 = dp_wrap(th2 + DP_TAU * w2);
            *nw2  = w2  + DP_TAU * w2d;

            // Reward shaping (mirror of the double-cartpole version, minus
            // the cart terms).
            float c1n  = cosf(*nth1);
            float c2n  = cosf(*nth2);
            float c12n = cosf(*nth1 - *nth2);
            float vw1  = *nw1;
            float vw2  = *nw2;

            float KE = 0.5f * m12    * DP_L1 * DP_L1 * vw1 * vw1
                     + 0.5f * DP_M2  * DP_L2 * DP_L2 * vw2 * vw2
                     +        DP_M2  * DP_L1 * DP_L2 * vw1 * vw2 * c12n;
            float PE = m12   * DP_G * DP_L1 * c1n
                     + DP_M2 * DP_G * DP_L2 * c2n;

            // Asymmetric energy penalty -- under-energy hurts 1.5x more than
            // over-energy. Pushes the agent past the mid-energy plateau
            // without rewarding runaway pumping.
            float E_diff = (KE + PE) - DP_E_TARGET;
            float E_err  = (E_diff < 0.0f)
                         ? 1.5f * (-E_diff) / (2.0f * DP_E_TARGET)
                         :         E_diff   / (2.0f * DP_E_TARGET);

            float g1 = fmaxf(0.0f, c1n);
            float g2 = fmaxf(0.0f, c2n);
            float upright_gate = g1 * g2;

            // Anti-alignment penalty -- punishes the "I shape" attractor
            // where link1 hangs down and link2 doubles back up.
            float align_diff = c1n - c2n;
            float align_pen  = 0.5f * align_diff * align_diff;

            // Velocity damping ONLY in the upright zone.
            float vel_pen = 0.1f * upright_gate * (vw1 * vw1 + vw2 * vw2);

            // Quadratic gate bonus -- steep near perfect upright.
            float gate2 = upright_gate * upright_gate;
            float bonus = 4.0f * gate2;

            // "Deep stillness" SMOOTH bowl -- replaces the previous hard
            // cliff. Continuous gradient pushes the policy to (a) sharpen
            // gate toward 1, AND (b) drive total angular velocity toward 0.
            //   peak: 5.0 at gate=1, w_total=0
            //   half: ~2.5 at gate~0.95, w_total~0.7
            //   fade: 0 once w_total >= 2.5 (stillness mask saturates)
            // Smooth everywhere -> no boundary to game.
            float w_total       = vw1 * vw1 + vw2 * vw2;
            float stillness     = fmaxf(0.0f, 1.0f - w_total / 2.5f);
            float stillness2    = stillness * stillness;
            float gate4         = gate2 * gate2;
            float deep          = 5.0f * gate4 * stillness2;

            *reward = 0.5f
                    + 0.5f * (c1n + c2n)        // cosine gradient
                    + bonus                      // quadratic upright bonus
                    + deep                       // stillness cliff
                    - 1.0f * E_err
                    - align_pen
                    - vel_pen;

            // No physical termination -- the angles wrap and omegas are
            // clipped via the grid bounds during interpolation.
            *terminated = false;
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        # No terminal state -- never absorbs. Returning an empty mask + 0.0.
        mask = np.zeros(len(states), dtype=bool)
        return mask, 0.0


# Python step (mirrors CUDA exactly, used at inference)
# --------------------------------------------------------------------------

_M1, _M2, _L1, _L2, _G, _TAU = 0.1, 0.1, 0.5, 0.5, 9.8, 0.02
_E_TARGET = (_M1 + _M2) * _G * _L1 + _M2 * _G * _L2  # ~1.47 J


def _step_python(state: np.ndarray, u: float):
    """Single Euler step matching the CUDA dynamics."""
    th1, w1, th2, w2 = state
    m12 = _M1 + _M2
    dth = th1 - th2
    c12 = np.cos(dth)
    s12 = np.sin(dth)

    M11 = m12 * _L1**2
    M12 = _M2 * _L1 * _L2 * c12
    M22 = _M2 * _L2**2

    b1 = u + m12 * _G * _L1 * np.sin(th1) - _M2 * _L1 * _L2 * s12 * w2**2
    b2 =     _M2 * _G * _L2 * np.sin(th2) + _M2 * _L1 * _L2 * s12 * w1**2

    det = M11 * M22 - M12 * M12
    w1d = (M22 * b1 - M12 * b2) / det
    w2d = (M11 * b2 - M12 * b1) / det

    nth1 = ((th1 + _TAU * w1 + np.pi) % (2 * np.pi)) - np.pi
    nw1  = w1 + _TAU * w1d
    nth2 = ((th2 + _TAU * w2 + np.pi) % (2 * np.pi)) - np.pi
    nw2  = w2 + _TAU * w2d

    next_state = np.array([nth1, nw1, nth2, nw2], dtype=np.float32)

    # Reward (must match CUDA exactly)
    c1n, c2n = np.cos(nth1), np.cos(nth2)
    c12n     = np.cos(nth1 - nth2)
    KE = 0.5 * m12 * _L1**2 * nw1**2 \
       + 0.5 * _M2 * _L2**2 * nw2**2 \
       +       _M2 * _L1 * _L2 * nw1 * nw2 * c12n
    PE = m12 * _G * _L1 * c1n + _M2 * _G * _L2 * c2n

    E_diff = (KE + PE) - _E_TARGET
    if E_diff < 0.0:
        E_err = 1.5 * (-E_diff) / (2.0 * _E_TARGET)
    else:
        E_err =        E_diff   / (2.0 * _E_TARGET)

    g1 = max(0.0, c1n)
    g2 = max(0.0, c2n)
    upright_gate = g1 * g2
    align_pen    = 0.5 * (c1n - c2n) ** 2
    vel_pen      = 0.1 * upright_gate * (nw1**2 + nw2**2)
    bonus        = 4.0 * upright_gate ** 2
    w_total      = nw1**2 + nw2**2
    stillness    = max(0.0, 1.0 - w_total / 2.5)
    deep         = 5.0 * (upright_gate ** 4) * (stillness ** 2)

    reward = 0.5 \
           + 0.5 * (c1n + c2n) \
           + bonus \
           + deep \
           - 1.0 * E_err \
           - align_pen \
           - vel_pen
    return next_state, reward, False


def _pole_energy(state: np.ndarray) -> float:
    """KE_rot + PE of the two-link pendulum (point masses at tips)."""
    _, w1, _, w2 = state
    th1, _, th2, _ = state
    KE = 0.5 * (_M1 + _M2) * _L1**2 * w1**2 \
       + 0.5 * _M2         * _L2**2 * w2**2 \
       +       _M2 * _L1 * _L2 * w1 * w2 * np.cos(th1 - th2)
    PE = (_M1 + _M2) * _G * _L1 * np.cos(th1) + _M2 * _G * _L2 * np.cos(th2)
    return float(KE + PE)


# Training
# --------------------------------------------------------------------------

def train(
    save_path: Path = Path("results/double_pendulum_swingup_cuda_policy.npz"),
) -> DoublePendulumSwingUpCuda:
    config = CudaPIConfig(
        gamma         = 0.999,
        theta         = 1e-4,
        max_eval_iter = 15_000,
        max_pi_iter   = 300,
        log_interval  = 500,
    )
    pi = DoublePendulumSwingUpCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# Rendering (pygame)
# --------------------------------------------------------------------------

_SCREEN_W   = 600
_SCREEN_H   = 600
_RENDER_FPS = 50


def _render_frame_pygame(screen, clock, state):
    """Two-link pendulum on a fixed pivot. theta=0 -> upright (top)."""
    import pygame
    from pygame import gfxdraw

    th1, _, th2, _ = state

    pivot_x = _SCREEN_W // 2
    pivot_y = _SCREEN_H // 2
    scale   = 200  # px per metre -> link of 0.5 m = 100 px

    L1 = scale * _L1
    L2 = scale * _L2

    surf = pygame.Surface((_SCREEN_W, _SCREEN_H))
    surf.fill((255, 255, 255))

    # Horizontal reference line through the pivot.
    gfxdraw.hline(surf, 0, _SCREEN_W, pivot_y, (220, 220, 220))

    # Link 1 -- a thin rectangle anchored at pivot, rotated by -th1.
    pole_w = 8.0
    coords1 = [(-pole_w / 2, -pole_w / 2),
               (-pole_w / 2,  L1 - pole_w / 2),
               ( pole_w / 2,  L1 - pole_w / 2),
               ( pole_w / 2, -pole_w / 2)]
    p1 = []
    for c in coords1:
        v = pygame.math.Vector2(c).rotate_rad(-th1)
        p1.append((v.x + pivot_x, v.y + pivot_y))
    gfxdraw.aapolygon(surf, p1, (70, 130, 180))
    gfxdraw.filled_polygon(surf, p1, (70, 130, 180))

    # Joint between link 1 and link 2 -- the tip of link 1.
    tip1 = pygame.math.Vector2(0, L1).rotate_rad(-th1)
    joint_x = tip1.x + pivot_x
    joint_y = tip1.y + pivot_y

    # Link 2 -- anchored at the tip of link 1, rotated by -th2.
    coords2 = [(-pole_w / 2, -pole_w / 2),
               (-pole_w / 2,  L2 - pole_w / 2),
               ( pole_w / 2,  L2 - pole_w / 2),
               ( pole_w / 2, -pole_w / 2)]
    p2 = []
    for c in coords2:
        v = pygame.math.Vector2(c).rotate_rad(-th2)
        p2.append((v.x + joint_x, v.y + joint_y))
    gfxdraw.aapolygon(surf, p2, (220, 120, 50))
    gfxdraw.filled_polygon(surf, p2, (220, 120, 50))

    # Pivot marker.
    gfxdraw.aacircle(surf, pivot_x, pivot_y, 6, (60, 60, 60))
    gfxdraw.filled_circle(surf, pivot_x, pivot_y, 6, (60, 60, 60))

    # Joint marker.
    gfxdraw.aacircle(surf, int(joint_x), int(joint_y), 5, (129, 132, 203))
    gfxdraw.filled_circle(surf, int(joint_x), int(joint_y), 5, (129, 132, 203))

    # Vertical flip so theta=0 points UP (physics convention).
    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    # Info overlay (drawn after flip so it stays at the top).
    font = pygame.font.SysFont("monospace", 14)
    lines = [
        f"th1   : {np.degrees(state[0]):+7.1f} deg   w1 : {state[1]:+.2f}",
        f"th2   : {np.degrees(state[2]):+7.1f} deg   w2 : {state[3]:+.2f}",
        f"E/E_t : {_pole_energy(state) / _E_TARGET:+.3f}",
    ]
    for i, txt in enumerate(lines):
        screen.blit(font.render(txt, True, (60, 60, 60)), (8, 8 + i * 18))

    pygame.event.pump()
    clock.tick(_RENDER_FPS)
    pygame.display.flip()


# Evaluation
# --------------------------------------------------------------------------

def evaluate(
    pi: DoublePendulumSwingUpCuda,
    n_episodes: int = 3,
    render: bool = False,
    record_path: Path = None,
    seed: int = 42,
    max_steps: int = 1000,
) -> None:
    from utils.barycentric import get_optimal_action

    recording = record_path is not None
    do_render = render or recording
    rng = np.random.default_rng(seed=seed)
    all_frames = []

    if do_render:
        import pygame
        pygame.init()
        pygame.display.init()
        flags = 0 if render else pygame.NOFRAME
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H), flags)
        pygame.display.set_caption("Double Pendulum Swing-Up - CUDA Policy Iteration")
        clock = pygame.time.Clock()

    STATS_WINDOW = 100
    all_steady_windows = []

    for ep in range(n_episodes):
        # Both links hanging down + small perturbation.
        state = np.array([
            np.pi + rng.uniform(-0.05, 0.05),  # th1 ~ pi (hanging)
            0.0,                                # w1
            np.pi + rng.uniform(-0.05, 0.05),  # th2 ~ pi (hanging)
            0.0,                                # w2
        ], dtype=np.float32)
        state[0] = ((state[0] + np.pi) % (2 * np.pi)) - np.pi
        state[2] = ((state[2] + np.pi) % (2 * np.pi)) - np.pi

        total_reward = 0.0
        traj         = [state.copy()]

        for step in range(max_steps):
            if do_render:
                _render_frame_pygame(screen, clock, state)
                if recording:
                    frame = pygame.surfarray.array3d(screen)
                    all_frames.append(frame.transpose(1, 0, 2))

            u = float(get_optimal_action(
                state,
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))
            state, reward, _ = _step_python(state, u)
            total_reward += reward
            traj.append(state.copy())

        print(
            f"Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.1f}"
        )
        print(
            f"  final state:"
            f"  th1={np.degrees(state[0]):+6.1f} deg  w1={state[1]:+.2f}"
            f"  th2={np.degrees(state[2]):+6.1f} deg  w2={state[3]:+.2f}"
        )

        traj_arr = np.asarray(traj[-STATS_WINDOW:])
        all_steady_windows.append(traj_arr)
        cos1 = np.cos(traj_arr[:, 0])
        cos2 = np.cos(traj_arr[:, 2])
        w1a  = np.abs(traj_arr[:, 1])
        w2a  = np.abs(traj_arr[:, 3])
        E    = np.array([_pole_energy(s) for s in traj_arr])

        E_diff = E - _E_TARGET
        E_err  = np.where(E_diff < 0.0,
                          1.5 * np.abs(E_diff) / (2.0 * _E_TARGET),
                                    E_diff    / (2.0 * _E_TARGET))

        g1_arr     = np.maximum(0.0, cos1)
        g2_arr     = np.maximum(0.0, cos2)
        gate_arr   = g1_arr * g2_arr
        w_total    = traj_arr[:, 1]**2 + traj_arr[:, 3]**2
        stillness  = np.maximum(0.0, 1.0 - w_total / 2.5)
        deep_score = (gate_arr ** 4) * (stillness ** 2)
        deep_high  = ((cos1 > 0.95) & (cos2 > 0.95) & (w_total < 1.0)).astype(float)

        base_part  = 0.5
        cos_part   = (0.5 * (cos1 + cos2)).mean()
        bonus_part = 4.0 * (gate_arr ** 2).mean()
        deep_part  = 5.0 * deep_score.mean()
        e_part     = -1.0 * E_err.mean()
        align_part = -0.5 * ((cos1 - cos2) ** 2).mean()
        vel_part   = -0.1 * (gate_arr * w_total).mean()

        print(
            f"  last {len(traj_arr)} steps:"
            f"  <cos th1>={cos1.mean():+.3f}  <cos th2>={cos2.mean():+.3f}"
            f"  <|w1|>={w1a.mean():.2f}  <|w2|>={w2a.mean():.2f}"
            f"  high_frac={deep_high.mean():.2f}"
        )
        print(
            f"             "
            f"  <E/E_tgt>={E.mean()/_E_TARGET:+.3f}  <E_err>={E_err.mean():.3f}"
            f"  reward parts: base={base_part:+.2f}  cos={cos_part:+.3f}"
            f"  bonus={bonus_part:+.3f}  deep={deep_part:+.3f}"
            f"  align={align_part:+.3f}  E={e_part:+.3f}  vel={vel_part:+.3f}"
        )

    # Dump aggregate steady-state windows for autoresearch scoring.
    if all_steady_windows:
        traj_path = Path("results") / "last_trajectory_dp.npz"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        # Map 4D state to 6D-shaped layout used by eval_metric:
        # eval_metric expects [x, x_dot, th1, w1, th2, w2].
        # We pad x and x_dot with zeros so the same eval_metric works.
        traj_full = np.concatenate(all_steady_windows, axis=0)
        traj_6d   = np.zeros((traj_full.shape[0], 6), dtype=np.float32)
        traj_6d[:, 2:6] = traj_full     # th1, w1, th2, w2
        np.savez(
            traj_path,
            traj=traj_6d,
            n_episodes=n_episodes,
            steps_per_episode=STATS_WINDOW,
        )

    if do_render:
        pygame.quit()

    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.suffix.lower() == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=50, loop=0)
        else:
            imageio.mimsave(str(record_path), all_frames, fps=50,
                            macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


def evaluate_random(
    n_episodes: int = 3,
    seed: int = 42,
    max_steps: int = 500,
    render: bool = False,
    record_path: Path = None,
) -> None:
    """Random policy rollout -- physics sanity check, no training."""
    rng = np.random.default_rng(seed=seed)

    recording = record_path is not None
    do_render = render or recording
    all_frames = []

    if do_render:
        import pygame
        pygame.init()
        pygame.display.init()
        flags = 0 if render else pygame.NOFRAME
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H), flags)
        pygame.display.set_caption("Double Pendulum Swing-Up - Random")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        state = np.array([np.pi, 0.0, np.pi, 0.0], dtype=np.float32)
        total_reward = 0.0
        for step in range(max_steps):
            if do_render:
                _render_frame_pygame(screen, clock, state)
                if recording:
                    frame = pygame.surfarray.array3d(screen)
                    all_frames.append(frame.transpose(1, 0, 2))
            u = float(rng.choice(ACTION_SPACE))
            state, reward, _ = _step_python(state, u)
            total_reward += reward
        print(f"[random] Episode {ep + 1}: {step + 1} steps | "
              f"reward = {total_reward:.1f}")

    if do_render:
        pygame.quit()
    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.suffix.lower() == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=50, loop=0)
        else:
            imageio.mimsave(str(record_path), all_frames, fps=50,
                            macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


# Visualization
# --------------------------------------------------------------------------

def plot_value_slice(
    pi: DoublePendulumSwingUpCuda,
    save_path: Path = Path("results/double_pendulum_swingup_cuda_vf_slice.png"),
) -> None:
    """V(theta1, theta2) at w1 = w2 = 0."""
    th1_bins = np.unique(pi.states_space[:, 0])
    w1_bins  = np.unique(pi.states_space[:, 1])
    th2_bins = np.unique(pi.states_space[:, 2])
    w2_bins  = np.unique(pi.states_space[:, 3])

    n0, n1, n2, n3 = len(th1_bins), len(w1_bins), len(th2_bins), len(w2_bins)
    V4 = pi.value_function.reshape(n0, n1, n2, n3)

    iw1 = int(np.argmin(np.abs(w1_bins)))
    iw2 = int(np.argmin(np.abs(w2_bins)))
    V2  = V4[:, iw1, :, iw2]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.pcolormesh(
        np.degrees(th2_bins), np.degrees(th1_bins), V2,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Value")
    ax.axvline(0,  color="white", ls="--", lw=1.5)
    ax.axhline(0,  color="white", ls="--", lw=1.5)
    ax.axvline( 180, color="red", ls=":", lw=1)
    ax.axvline(-180, color="red", ls=":", lw=1)
    ax.axhline( 180, color="red", ls=":", lw=1)
    ax.axhline(-180, color="red", ls=":", lw=1)
    ax.set_xlabel("theta2 (deg)")
    ax.set_ylabel("theta1 (deg)")
    ax.set_title("Double Pendulum Swing-Up - V(th1, th2) at w1=w2=0")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value slice saved to {save_path.resolve()}")


def plot_trajectory(
    pi: DoublePendulumSwingUpCuda,
    save_path: Path = Path("results/double_pendulum_swingup_cuda_trajectory.png"),
    seed: int = 42,
    max_steps: int = 1000,
) -> None:
    """Multi-panel trajectory from a hanging-down start."""
    from utils.barycentric import get_optimal_action

    rng    = np.random.default_rng(seed=seed)
    th1_0  = ((np.pi + rng.uniform(-0.05, 0.05) + np.pi) % (2 * np.pi)) - np.pi
    th2_0  = ((np.pi + rng.uniform(-0.05, 0.05) + np.pi) % (2 * np.pi)) - np.pi
    state  = np.array([th1_0, 0.0, th2_0, 0.0], dtype=np.float32)

    th1s, th2s, w1s, w2s, rs, Es = [], [], [], [], [], []

    for _ in range(max_steps):
        u = float(get_optimal_action(
            state, pi.policy, pi.action_space,
            pi.bounds_low, pi.bounds_high,
            pi.grid_shape, pi.strides, pi.corner_bits,
        ))
        state, reward, _ = _step_python(state, u)
        th1s.append(np.degrees(state[0]))
        th2s.append(np.degrees(state[2]))
        w1s.append(state[1])
        w2s.append(state[3])
        rs.append(reward)
        Es.append(_pole_energy(state) / _E_TARGET)

    t = np.arange(len(rs)) * _TAU

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t, th1s, color="steelblue", label="theta1")
    axes[0].plot(t, th2s, color="darkorange", label="theta2")
    axes[0].axhline(  0, color="green", ls="--", lw=1, label="upright")
    axes[0].axhline( 180, color="red",  ls=":",  lw=1, label="hanging")
    axes[0].axhline(-180, color="red",  ls=":",  lw=1)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("Double Pendulum Swing-Up - trajectory from hanging-down")

    axes[1].plot(t, w1s, color="steelblue",  label="w1")
    axes[1].plot(t, w2s, color="darkorange", label="w2")
    axes[1].axhline(0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("Angular vel (rad/s)")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(t, Es, color="purple")
    axes[2].axhline(1.0,  color="green", ls="--", lw=1, label="E_target")
    axes[2].axhline(-1.0, color="red",   ls=":",  lw=1, label="hanging rest")
    axes[2].set_ylabel("E_pole / E_target")
    axes[2].legend(loc="upper right", fontsize=9)

    axes[3].plot(t, rs, color="purple")
    axes[3].axhline(5.0, color="green", ls="--", lw=1, label="max reward")
    axes[3].axhline(0.0, color="gray",  ls="--", lw=1)
    axes[3].set_ylabel("Reward")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to {save_path.resolve()}")


# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Double Pendulum Swing-Up - CUDA Policy Iteration"
    )
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes with pygame")
    parser.add_argument("--random",    type=int, nargs="?", const=3, default=None,
                        metavar="N",
                        help="Run N random-policy episodes (physics check)")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save evaluation video to PATH (.gif or .mp4)")
    parser.add_argument("--episodes",  type=int, default=3,
                        help="Number of evaluation episodes (default: 3)")
    parser.add_argument("--steps",     type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--bins",      type=int, default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM}). "
                             "Memory: 15->~50K states, 25->~390K, 30->~810K")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip saving plots")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path,
                        default=Path("results/double_pendulum_swingup_cuda_policy.npz"))
    args = parser.parse_args()

    if args.random is not None:
        evaluate_random(
            n_episodes=args.random,
            seed=args.seed,
            max_steps=args.steps,
            render=args.render,
            record_path=args.record,
        )
    else:
        if args.bins != BINS_PER_DIM:
            for key in BINS_SPACE:
                lo, hi = BINS_SPACE[key][0], BINS_SPACE[key][-1]
                BINS_SPACE[key] = np.linspace(lo, hi, args.bins, dtype=np.float32)

        if args.save_path.exists() and not args.retrain:
            print(f"[+] Loading existing policy from {args.save_path}")
            pi = DoublePendulumSwingUpCuda.load(args.save_path)
        else:
            print("[*] Training new policy...")
            pi = train(args.save_path)

        evaluate(pi, n_episodes=args.episodes, render=args.render,
                 record_path=args.record, seed=args.seed,
                 max_steps=args.steps)
        if not args.no_plot:
            plot_value_slice(pi)
            plot_trajectory(pi, seed=args.seed, max_steps=args.steps)
