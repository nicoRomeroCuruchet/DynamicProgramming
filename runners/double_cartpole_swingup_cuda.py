"""
runners/double_cartpole_swingup_cuda.py — CUDA Policy Iteration for Double CartPole Swing-Up.

Same Lagrangian dynamics as double_cartpole_cuda.py, but now the task is to swing
BOTH poles up from the hanging-down position (theta = pi) and balance them at the
upright equilibrium (theta = 0).

State  : [x        ∈ [-2.5,  2.5]   cart position (terminates at |x| > 2.4)
           x_dot   ∈ [-6.0,  6.0]   cart velocity
           theta1  ∈ [-pi,   pi]    first pole angle from vertical (full range)
           th1_dot ∈ [-15,   15]    first pole angular velocity
           theta2  ∈ [-pi,   pi]    second pole angle from vertical (full range)
           th2_dot ∈ [-15,   15]    second pole angular velocity]

Termination: |x| > 2.4 only — angles are free to spin through full [-pi, pi]
             and are wrapped back after each Euler step.

Actions: 5 force values {-30, -15, 0, +15, +30} applied to the cart.
           More authority is needed to pump enough energy into a double
           pendulum than into a single one.

Reward : 0.5*(cos th1 + cos th2)
         - 0.5 * E_err
         - 0.1 * (x/2.4)^2
         - 0.001 * gate * (w1^2 + w2^2)

           E_err  = |E_pole - E_target| / (2 * E_target)
           E_pole = KE_rot + PE of the two-link pendulum
           E_target = (m1+m2)*g*l1 + m2*g*l2
           gate   = max(0, 0.5*(cos th1 + cos th2))   (0 hanging, 1 upright)

           The energy term breaks the flatness of cos-only reward at
           the hanging position, giving DP a gradient in (w1, w2) for
           swing-up. The upright-gated velocity penalty kills the
           false attractor where the first link stays upright while
           the second link free-spins at E ≈ E_target — without it
           the policy is happy to leave the second pole rotating
           because cos(theta2) averages to ~0 there but E_err ≈ 0.

Memory (BINS_PER_DIM bins per dimension):
  BINS_PER_DIM=15 → 15^6 = 11.4M states  → ~420 MB VRAM
  BINS_PER_DIM=20 → 20^6 = 64.0M states  → ~2.1 GB VRAM

Dynamics: identical to double_cartpole_cuda.py (Euler, Lagrangian 3x3 mass matrix).
  Cart mass M=1.0, pole masses m1=m2=0.1, pole lengths l1=l2=0.5, tau=0.02.
  Angles are wrapped to (-pi, pi] after each Euler step.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cuda_policy_iteration import CudaPolicyIteration6D, CudaPIConfig


# ── Grid & action space ────────────────────────────────────────────────────────

BINS_PER_DIM = 20   # 20^6 ~ 64M states ~ 2.4 GB VRAM

BINS_SPACE = {
    "x":       np.linspace(-2.5,       2.5,       BINS_PER_DIM, dtype=np.float32),
    "x_dot":   np.linspace(-8.0,       8.0,       BINS_PER_DIM, dtype=np.float32),
    "theta1":  np.linspace(-np.pi,     np.pi,     BINS_PER_DIM, dtype=np.float32),
    "th1_dot": np.linspace(-15.0,      15.0,      BINS_PER_DIM, dtype=np.float32),
    "theta2":  np.linspace(-np.pi,     np.pi,     BINS_PER_DIM, dtype=np.float32),
    "th2_dot": np.linspace(-15.0,      15.0,      BINS_PER_DIM, dtype=np.float32),
}

# Five actions give the policy the option to do nothing during the swing
# phase (mirrors the single-pole swing-up). Wider authority is needed
# for the heavier two-link pendulum.
ACTION_SPACE = np.array([-60.0, -30.0, -10.0, 0.0, 10.0, 30.0, 60.0], dtype=np.float32)


# ── CUDA subclass ──────────────────────────────────────────────────────────────

class DoubleCartPoleSwingUpCuda(CudaPolicyIteration6D):
    """
    CudaPolicyIteration6D for double inverted pendulum swing-up.

    State: [x, x_dot, theta1, th1_dot, theta2, th2_dot]
    Both angles are absolute from vertical: 0 = upright (goal), pi = hanging down (start).
    The second pole is attached to the tip of the first.
    Angles are wrapped to (-pi, pi] after each Euler step.
    """

    def _dynamics_cuda_src(self) -> str:
        return r'''
        // --- Physical constants -------------------------------------------
        #define DCP_G     9.8f
        #define DCP_M     1.0f    // cart mass
        #define DCP_M1    0.1f    // first pole mass (at tip)
        #define DCP_M2    0.1f    // second pole mass (at tip)
        #define DCP_L1    0.5f    // first pole length (mass at the tip)
        #define DCP_L2    0.5f    // second pole length (mass at the tip)
        #define DCP_TAU   0.02f
        #define DCP_XMAX  2.4f
        #define DCP_PI    3.14159265358979323846f
        // Total mechanical energy of the two-link pendulum at upright rest:
        //   PE = (m1+m2)*g*l1 + m2*g*l2,  KE = 0
        // Used to build a reward gradient in the angular velocities at
        // the hanging-down rest position (where cos-only reward is flat).
        #define DCP_E_TARGET ((DCP_M1 + DCP_M2) * DCP_G * DCP_L1 \
                            + DCP_M2 * DCP_G * DCP_L2)

        __device__ float dcp_wrap(float x) {
            float r = fmodf(x + DCP_PI, 2.0f * DCP_PI);
            if (r < 0.0f) r += 2.0f * DCP_PI;
            return r - DCP_PI;
        }

        __device__ void step_dynamics(
            float x,   float xd,
            float th1, float th1d,
            float th2, float th2d,
            float force,
            float* nx,   float* nxd,
            float* nth1, float* nth1d,
            float* nth2, float* nth2d,
            float* reward, bool* terminated
        ) {
            float m12   = DCP_M1 + DCP_M2;
            float cos1  = cosf(th1), sin1  = sinf(th1);
            float cos2  = cosf(th2), sin2  = sinf(th2);
            float dth   = th1 - th2;
            float cos12 = cosf(dth), sin12 = sinf(dth);

            // --- Mass matrix elements (symmetric 3x3) ---------------------
            float a = DCP_M + m12;
            float b = m12   * DCP_L1 * cos1;
            float c = DCP_M2 * DCP_L2 * cos2;
            float d = m12   * DCP_L1 * DCP_L1;
            float e = DCP_M2 * DCP_L1 * DCP_L2 * cos12;
            float f = DCP_M2 * DCP_L2 * DCP_L2;

            // --- Generalized forces + centrifugal/Coriolis ----------------
            float rhs1 = force
                       + m12    * DCP_L1 * th1d * th1d * sin1
                       + DCP_M2 * DCP_L2 * th2d * th2d * sin2;
            float rhs2 = m12    * DCP_G  * DCP_L1 * sin1
                       - DCP_M2 * DCP_L1 * DCP_L2 * th2d * th2d * sin12;
            float rhs3 = DCP_M2 * DCP_G  * DCP_L2 * sin2
                       + DCP_M2 * DCP_L1 * DCP_L2 * th1d * th1d * sin12;

            // --- Analytic 3x3 inverse via cofactors -----------------------
            float cof11 = d * f - e * e;
            float cof12 = e * c - b * f;
            float cof13 = b * e - d * c;
            float cof22 = a * f - c * c;
            float cof23 = b * c - a * e;
            float cof33 = a * d - b * b;

            float det     = a * cof11 + b * cof12 + c * cof13;
            float inv_det = 1.0f / det;

            float xacc   = (cof11 * rhs1 + cof12 * rhs2 + cof13 * rhs3) * inv_det;
            float th1acc = (cof12 * rhs1 + cof22 * rhs2 + cof23 * rhs3) * inv_det;
            float th2acc = (cof13 * rhs1 + cof23 * rhs2 + cof33 * rhs3) * inv_det;

            // --- Euler integration + angle wrap ---------------------------
            *nx    = x    + DCP_TAU * xd;
            *nxd   = xd   + DCP_TAU * xacc;
            *nth1  = dcp_wrap(th1  + DCP_TAU * th1d);
            *nth1d = th1d + DCP_TAU * th1acc;
            *nth2  = dcp_wrap(th2  + DCP_TAU * th2d);
            *nth2d = th2d + DCP_TAU * th2acc;

            // --- Reward = cosine + energy shaping -------------------------
            // KE of the two-link pendulum (rotational, in the cart frame):
            //   KE = 0.5*(m1+m2)*l1^2*w1^2 + 0.5*m2*l2^2*w2^2
            //      + m2*l1*l2*w1*w2*cos(th1 - th2)
            // PE (with reference at the cart axle, +y = up):
            //   PE = (m1+m2)*g*l1*cos(th1) + m2*g*l2*cos(th2)
            // E_err is small only near the upright energy level.
            // At hanging rest E = -E_target -> E_err = 1 (saturated).
            // At hanging with the right pump-up speed E ~ +E_target -> E_err = 0.
            // This gives a gradient in (w1, w2) at the bottom that drives
            // the swing-up phase, which a cos-only reward does not.
            float c1n  = cosf(*nth1);
            float c2n  = cosf(*nth2);
            float c12n = cosf(*nth1 - *nth2);
            float w1   = *nth1d;
            float w2   = *nth2d;

            float KE = 0.5f * m12    * DCP_L1 * DCP_L1 * w1 * w1
                     + 0.5f * DCP_M2 * DCP_L2 * DCP_L2 * w2 * w2
                     +        DCP_M2 * DCP_L1 * DCP_L2 * w1 * w2 * c12n;
            float PE = m12    * DCP_G * DCP_L1 * c1n
                     + DCP_M2 * DCP_G * DCP_L2 * c2n;

            // UNCAPPED energy error -- saturating let mid-energy plateau
            // be "good enough", trapping the agent at <E/E_tgt> ~ 0.4.
            float E_err = fabsf((KE + PE) - DCP_E_TARGET)
                        / (2.0f * DCP_E_TARGET);

            // Per-link upright "softgates" -- give partial credit as each
            // link approaches upright, breaking the symmetric plateau.
            float g1 = fmaxf(0.0f, c1n);
            float g2 = fmaxf(0.0f, c2n);
            float upright_gate = g1 * g2;

            // Velocity damping ONLY in the upright zone.
            float vel_pen = 0.05f * upright_gate * (w1 * w1 + w2 * w2);

            // Cart velocity penalty -- discourages "fly to the wall".
            float xdn = *nxd / 8.0f;
            float xn  = *nx  / DCP_XMAX;

            // +1.0 baseline so surviving beats terminating early.
            *reward = 1.0f
                    + 0.5f * (c1n + c2n)        // base cosine,  [-1, +1]
                    + 0.5f * g1                 // per-link credit (link 1)
                    + 1.0f * g2                 // per-link credit (link 2 - harder)
                    + 2.0f * upright_gate       // BIG bonus when both up
                    - 1.0f * E_err              // uncapped energy penalty
                    - 0.5f * xn * xn
                    - 0.2f * xdn * xdn
                    - vel_pen;

            if ((*nx < -DCP_XMAX) || (*nx > DCP_XMAX)) {
                *reward -= 100.0f;
            }

            // Only terminate on cart bounds -- angles are free to spin
            *terminated = (*nx < -DCP_XMAX) || (*nx > DCP_XMAX);
        }
        '''

    def _terminal_fn(self, states: np.ndarray):
        """Cart-out-of-bounds is the only terminal condition."""
        x    = states[:, 0]
        mask = (x < -2.4) | (x > 2.4)
        return mask, 0.0


# ── Python step function (for evaluate — no gymnasium env exists) ─────────────

def _step_python(state, force):
    """Single Euler step matching the CUDA dynamics — used at inference."""
    x, xd, th1, th1d, th2, th2d = state
    M, m1, m2, l1, l2 = 1.0, 0.1, 0.1, 0.5, 0.5
    g, tau = 9.8, 0.02

    m12   = m1 + m2
    cos1  = np.cos(th1); sin1  = np.sin(th1)
    cos2  = np.cos(th2); sin2  = np.sin(th2)
    dth   = th1 - th2
    cos12 = np.cos(dth); sin12 = np.sin(dth)

    a = M + m12
    b = m12 * l1 * cos1
    c = m2  * l2 * cos2
    d = m12 * l1 * l1
    e = m2  * l1 * l2 * cos12
    f = m2  * l2 * l2

    rhs1 = force + m12 * l1 * th1d**2 * sin1 + m2 * l2 * th2d**2 * sin2
    rhs2 = m12 * g * l1 * sin1 - m2 * l1 * l2 * th2d**2 * sin12
    rhs3 = m2  * g * l2 * sin2 + m2 * l1 * l2 * th1d**2 * sin12

    cof11 = d*f - e*e;  cof12 = e*c - b*f;  cof13 = b*e - d*c
    cof22 = a*f - c*c;  cof23 = b*c - a*e;  cof33 = a*d - b*b
    det   = a*cof11 + b*cof12 + c*cof13

    xacc   = (cof11*rhs1 + cof12*rhs2 + cof13*rhs3) / det
    th1acc = (cof12*rhs1 + cof22*rhs2 + cof23*rhs3) / det
    th2acc = (cof13*rhs1 + cof23*rhs2 + cof33*rhs3) / det

    nx    = x    + tau * xd
    nxd   = xd   + tau * xacc
    nth1  = ((th1  + tau * th1d + np.pi) % (2 * np.pi)) - np.pi
    nth1d = th1d + tau * th1acc
    nth2  = ((th2  + tau * th2d + np.pi) % (2 * np.pi)) - np.pi
    nth2d = th2d + tau * th2acc

    next_state = np.array([nx, nxd, nth1, nth1d, nth2, nth2d], dtype=np.float32)

    # Reward shaping — must match CUDA exactly.
    c1n, c2n = np.cos(nth1), np.cos(nth2)
    c12n     = np.cos(nth1 - nth2)
    KE = 0.5 * m12 * l1**2 * nth1d**2 \
       + 0.5 * m2  * l2**2 * nth2d**2 \
       +        m2 * l1 * l2 * nth1d * nth2d * c12n
    PE = m12 * 9.8 * l1 * c1n + m2 * 9.8 * l2 * c2n
    E_target = m12 * 9.8 * l1 + m2 * 9.8 * l2
    E_err    = abs((KE + PE) - E_target) / (2.0 * E_target)  # uncapped

    g1 = max(0.0, c1n)
    g2 = max(0.0, c2n)
    upright_gate = g1 * g2
    vel_pen      = 0.05 * upright_gate * (nth1d**2 + nth2d**2)

    xn  = nx  / 2.4
    xdn = nxd / 8.0
    reward = 1.0 \
           + 0.5 * (c1n + c2n) \
           + 0.5 * g1 \
           + 1.0 * g2 \
           + 2.0 * upright_gate \
           - 1.0 * E_err \
           - 0.5 * xn**2 \
           - 0.2 * xdn**2 \
           - vel_pen
    terminated = abs(nx) > 2.4
    if terminated:
        reward -= 100.0
    return next_state, reward, terminated


# ── Energy helpers (for diagnostics, must match CUDA reward) ──────────────────

_M1, _M2, _L1, _L2, _G = 0.1, 0.1, 0.5, 0.5, 9.8
_E_TARGET = (_M1 + _M2) * _G * _L1 + _M2 * _G * _L2   # ≈ 1.47 J


def _pole_energy(state: np.ndarray) -> float:
    """KE_rot + PE of the two-link pendulum (cart-frame, mass at tips)."""
    _, _, th1, w1, th2, w2 = state
    KE = 0.5 * (_M1 + _M2) * _L1**2 * w1**2 \
       + 0.5 * _M2         * _L2**2 * w2**2 \
       +        _M2 * _L1 * _L2 * w1 * w2 * np.cos(th1 - th2)
    PE = (_M1 + _M2) * _G * _L1 * np.cos(th1) + _M2 * _G * _L2 * np.cos(th2)
    return KE + PE


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    save_path: Path = Path("results/double_cartpole_swingup_cuda_policy.npz"),
) -> DoubleCartPoleSwingUpCuda:
    config = CudaPIConfig(
        gamma         = 0.999,
        theta         = 1e-4,
        max_eval_iter = 20_000,
        max_pi_iter   = 300,
        log_interval  = 500,
    )

    pi = DoubleCartPoleSwingUpCuda(BINS_SPACE, ACTION_SPACE, config)
    pi.run()
    pi.save(save_path)
    return pi


# ── Rendering (pygame) ────────────────────────────────────────────────────────

_SCREEN_W   = 600
_SCREEN_H   = 400
_RENDER_FPS = 50

def _render_frame_pygame(screen, clock, state):
    """Draw cart + double pendulum on the pygame surface."""
    import pygame
    from pygame import gfxdraw

    x, _, th1, _, th2, _ = state

    world_width = 2.4 * 2
    scale       = _SCREEN_W / world_width

    cart_width  = 50.0
    cart_height = 30.0
    axle_offset = cart_height / 4.0
    cart_y      = 200              # center of screen — poles can hang below

    pole_w  = 10.0
    pole_l  = scale * (2 * 0.5)   # 125 px
    pole2_w = 8.0

    cart_x = x * scale + _SCREEN_W / 2.0

    surf = pygame.Surface((_SCREEN_W, _SCREEN_H))
    surf.fill((255, 255, 255))

    # --- Ground line at center ---
    gfxdraw.hline(surf, 0, _SCREEN_W, cart_y, (180, 180, 180))

    # --- Cart ---
    l = -cart_width / 2;  r = cart_width / 2
    t =  cart_height / 2; b = -cart_height / 2
    cart_coords = [(c[0] + cart_x, c[1] + cart_y)
                   for c in [(l, b), (l, t), (r, t), (r, b)]]
    gfxdraw.aapolygon(surf, cart_coords, (0, 0, 0))
    gfxdraw.filled_polygon(surf, cart_coords, (0, 0, 0))

    axle_x = int(cart_x)
    axle_y = int(cart_y + axle_offset)

    # --- Pole 1 (blue) — drawn with rotate_rad(-th1) so that after the
    # vertical flip below, theta=0 displays the pole pointing UP. Same
    # convention used by the single-pole swing-up renderer.
    p1_coords = []
    for coord in [(-pole_w/2, -pole_w/2),
                  (-pole_w/2,  pole_l - pole_w/2),
                  ( pole_w/2,  pole_l - pole_w/2),
                  ( pole_w/2, -pole_w/2)]:
        v = pygame.math.Vector2(coord).rotate_rad(-th1)
        p1_coords.append((v.x + cart_x, v.y + cart_y + axle_offset))
    gfxdraw.aapolygon(surf, p1_coords, (70, 130, 180))
    gfxdraw.filled_polygon(surf, p1_coords, (70, 130, 180))

    gfxdraw.aacircle(surf, axle_x, axle_y, int(pole_w / 2), (129, 132, 203))
    gfxdraw.filled_circle(surf, axle_x, axle_y, int(pole_w / 2), (129, 132, 203))

    # --- Joint (tip of pole 1) — same sign convention as pole 1 ---
    tip1   = pygame.math.Vector2(0, pole_l).rotate_rad(-th1)
    joint_x = int(cart_x + tip1.x)
    joint_y = int(cart_y + axle_offset + tip1.y)

    # --- Pole 2 (orange) ---
    p2_coords = []
    for coord in [(-pole2_w/2, -pole2_w/2),
                  (-pole2_w/2,  pole_l - pole2_w/2),
                  ( pole2_w/2,  pole_l - pole2_w/2),
                  ( pole2_w/2, -pole2_w/2)]:
        v = pygame.math.Vector2(coord).rotate_rad(-th2)
        p2_coords.append((v.x + joint_x, v.y + joint_y))
    gfxdraw.aapolygon(surf, p2_coords, (210, 105, 30))
    gfxdraw.filled_polygon(surf, p2_coords, (210, 105, 30))

    gfxdraw.aacircle(surf, joint_x, joint_y, int(pole2_w / 2), (129, 132, 203))
    gfxdraw.filled_circle(surf, joint_x, joint_y, int(pole2_w / 2), (129, 132, 203))

    # Flip vertically: pygame y-down → visual y-up (theta=0 now points UP)
    surf = pygame.transform.flip(surf, False, True)
    screen.blit(surf, (0, 0))

    # Info overlay drawn directly on the screen (after flip) so the text
    # stays upright at the top.
    font = pygame.font.SysFont("monospace", 14)
    lines = [
        f"x    : {state[0]:+.3f} m",
        f"th1  : {np.degrees(state[2]):+.1f} deg",
        f"th2  : {np.degrees(state[4]):+.1f} deg",
    ]
    for i, txt in enumerate(lines):
        screen.blit(font.render(txt, True, (60, 60, 60)), (8, 8 + i * 18))

    pygame.event.pump()
    clock.tick(_RENDER_FPS)
    pygame.display.flip()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pi: DoubleCartPoleSwingUpCuda,
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
        pygame.display.set_caption("Double CartPole Swing-Up — CUDA Policy Iteration")
        clock = pygame.time.Clock()

    STATS_WINDOW = 100   # last-N steps used for steady-state diagnostics

    for ep in range(n_episodes):
        # Start with both poles hanging down (theta = pi) + small perturbation
        state = np.array([
            rng.uniform(-0.1,  0.1),    # x
            0.0,                         # x_dot
            np.pi + rng.uniform(-0.05, 0.05),  # th1 ≈ pi (hanging down)
            0.0,                         # th1_dot
            np.pi + rng.uniform(-0.05, 0.05),  # th2 ≈ pi (hanging down)
            0.0,                         # th2_dot
        ], dtype=np.float32)
        state[2] = ((state[2] + np.pi) % (2 * np.pi)) - np.pi
        state[4] = ((state[4] + np.pi) % (2 * np.pi)) - np.pi

        total_reward = 0.0
        terminated   = False
        traj         = [state.copy()]
        forces       = []

        for step in range(max_steps):
            if do_render:
                _render_frame_pygame(screen, clock, state)
                if recording:
                    frame = pygame.surfarray.array3d(screen)
                    all_frames.append(frame.transpose(1, 0, 2))

            force = float(get_optimal_action(
                state,
                pi.policy, pi.action_space,
                pi.bounds_low, pi.bounds_high,
                pi.grid_shape, pi.strides, pi.corner_bits,
            ))
            state, reward, terminated = _step_python(state, force)
            total_reward += reward
            traj.append(state.copy())
            forces.append(force)
            if terminated:
                break

        outcome = "cart out-of-bounds" if terminated else f"{max_steps} steps"
        print(
            f"Episode {ep + 1}: {step + 1} steps | "
            f"reward = {total_reward:.1f} | {outcome}"
        )
        print(
            f"  final state:"
            f"  x={state[0]:+.3f}  xd={state[1]:+.2f}"
            f"  th1={np.degrees(state[2]):+6.1f}°  w1={state[3]:+.2f}"
            f"  th2={np.degrees(state[4]):+6.1f}°  w2={state[5]:+.2f}"
        )

        # Steady-state diagnostics over the last STATS_WINDOW steps.
        traj_arr = np.asarray(traj[-STATS_WINDOW:])
        cos1   = np.cos(traj_arr[:, 2])
        cos2   = np.cos(traj_arr[:, 4])
        w1_abs = np.abs(traj_arr[:, 3])
        w2_abs = np.abs(traj_arr[:, 5])
        E      = np.array([_pole_energy(s) for s in traj_arr])
        E_err  = np.abs(E - _E_TARGET) / (2.0 * _E_TARGET)  # uncapped

        # Reward decomposition (matches the CUDA shape exactly).
        g1_arr   = np.maximum(0.0, cos1)
        g2_arr   = np.maximum(0.0, cos2)
        gate_arr = g1_arr * g2_arr
        base_part  = 1.0
        cos_part   = (0.5 * (cos1 + cos2)).mean()
        g1_part    = 0.5 * g1_arr.mean()
        g2_part    = 1.0 * g2_arr.mean()
        bonus_part = 2.0 * gate_arr.mean()
        e_part     = -1.0 * E_err.mean()
        x_part     = -0.5 * ((traj_arr[:, 0] / 2.4) ** 2).mean()
        xd_part    = -0.2 * ((traj_arr[:, 1] / 8.0) ** 2).mean()
        vel_part   = -0.05 * (gate_arr * (traj_arr[:, 3]**2 + traj_arr[:, 5]**2)).mean()

        print(
            f"  last {len(traj_arr)} steps:"
            f"  <cos th1>={cos1.mean():+.3f}  <cos th2>={cos2.mean():+.3f}"
            f"  <|w1|>={w1_abs.mean():.2f}  <|w2|>={w2_abs.mean():.2f}"
        )
        print(
            f"             "
            f"  <E/E_tgt>={E.mean()/_E_TARGET:+.3f}  <E_err>={E_err.mean():.3f}"
            f"  reward parts: base={base_part:+.2f}  cos={cos_part:+.3f}"
            f"  g1={g1_part:+.3f}  g2={g2_part:+.3f}  bonus={bonus_part:+.3f}"
            f"  E={e_part:+.3f}  x={x_part:+.3f}  xd={xd_part:+.3f}  vel={vel_part:+.3f}"
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
            imageio.mimsave(str(record_path), all_frames, fps=50, macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


def evaluate_random(
    n_episodes: int = 3,
    seed: int = 42,
    max_steps: int = 500,
    render: bool = False,
    record_path: Path = None,
) -> None:
    """Random policy rollout — physics sanity check, no training needed."""
    rng = np.random.default_rng(seed=seed)

    recording  = record_path is not None
    do_render  = render or recording
    all_frames = []

    if do_render:
        import pygame
        pygame.init()
        pygame.display.init()
        flags = 0 if render else pygame.NOFRAME
        screen = pygame.display.set_mode((_SCREEN_W, _SCREEN_H), flags)
        pygame.display.set_caption("Double CartPole Swing-Up — Random (physics check)")
        clock = pygame.time.Clock()

    for ep in range(n_episodes):
        state = np.array([0.0, 0.0, np.pi, 0.0, np.pi, 0.0], dtype=np.float32)
        total_reward = 0.0
        for step in range(max_steps):
            if do_render:
                _render_frame_pygame(screen, clock, state)
                if recording:
                    frame = pygame.surfarray.array3d(screen)
                    all_frames.append(frame.transpose(1, 0, 2))

            force = float(rng.choice(ACTION_SPACE))
            state, reward, terminated = _step_python(state, force)
            total_reward += reward
            if terminated:
                break
        print(f"[random] Episode {ep + 1}: {step + 1} steps | reward = {total_reward:.1f}")

    if do_render:
        pygame.quit()

    if recording and all_frames:
        import imageio
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if record_path.suffix.lower() == ".gif":
            imageio.mimsave(str(record_path), all_frames, fps=50, loop=0)
        else:
            imageio.mimsave(str(record_path), all_frames, fps=50, macro_block_size=1)
        print(f"Video saved to {record_path.resolve()}")


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_value_slice(
    pi: DoubleCartPoleSwingUpCuda,
    save_path: Path = Path("results/double_cartpole_swingup_cuda_vf_slice.png"),
) -> None:
    """
    V(theta1, theta2) heatmap at x=0, x_dot=0, th1_dot=0, th2_dot=0.
    With full [-pi, pi] range this shows the entire angle landscape.
    """
    bins  = [np.unique(pi.states_space[:, d]) for d in range(6)]
    shape = [len(b) for b in bins]
    V6    = pi.value_function.reshape(shape)

    ix   = int(np.argmin(np.abs(bins[0])))
    ixd  = int(np.argmin(np.abs(bins[1])))
    it1d = int(np.argmin(np.abs(bins[3])))
    it2d = int(np.argmin(np.abs(bins[5])))

    V2 = V6[ix, ixd, :, it1d, :, it2d]   # (n_th1, n_th2)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(
        np.degrees(bins[4]), np.degrees(bins[2]), V2,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Value")
    ax.axvline(0,   color="white", ls="--", lw=1, label="th2=0 (upright)")
    ax.axhline(0,   color="white", ls="--", lw=1, label="th1=0 (upright)")
    ax.axvline(180, color="red",   ls=":",  lw=1)
    ax.axvline(-180,color="red",   ls=":",  lw=1)
    ax.set_xlabel("theta2 (deg)")
    ax.set_ylabel("theta1 (deg)")
    ax.set_title("Double CartPole Swing-Up — V(th1, th2) at x=xd=w1=w2=0")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Value slice saved to {save_path.resolve()}")


def plot_trajectory(
    pi: DoubleCartPoleSwingUpCuda,
    save_path: Path = Path("results/double_cartpole_swingup_cuda_trajectory.png"),
    seed: int = 42,
    max_steps: int = 1000,
) -> None:
    """
    Trajectory rollout from hanging-down. Five panels:
      1. theta1, theta2  vs t          (angles in deg)
      2. w1, w2          vs t          (angular velocities, rad/s)
      3. x, x_dot        vs t          (cart kinematics)
      4. E_pole / E_target vs t        (energy ratio — should reach 1 and stay)
      5. reward          vs t          (total + decomposition)
    """
    from utils.barycentric import get_optimal_action

    rng   = np.random.default_rng(seed=seed)
    state = np.array([0.0, 0.0, np.pi + rng.uniform(-0.05, 0.05),
                      0.0, np.pi + rng.uniform(-0.05, 0.05), 0.0], dtype=np.float32)
    state[2] = ((state[2] + np.pi) % (2 * np.pi)) - np.pi
    state[4] = ((state[4] + np.pi) % (2 * np.pi)) - np.pi

    states  = [state.copy()]
    rewards = []

    for _ in range(max_steps):
        force = float(get_optimal_action(
            state, pi.policy, pi.action_space,
            pi.bounds_low, pi.bounds_high,
            pi.grid_shape, pi.strides, pi.corner_bits,
        ))
        state, reward, terminated = _step_python(state, force)
        states.append(state.copy())
        rewards.append(reward)
        if terminated:
            break

    S = np.asarray(states[1:])           # (T, 6)
    t = np.arange(len(S)) * 0.02

    xs, xds  = S[:, 0], S[:, 1]
    th1, w1  = np.degrees(S[:, 2]), S[:, 3]
    th2, w2  = np.degrees(S[:, 4]), S[:, 5]
    E        = np.array([_pole_energy(s) for s in S]) / _E_TARGET
    rs       = np.asarray(rewards)
    cos_part = 0.5 * (np.cos(S[:, 2]) + np.cos(S[:, 4]))
    x_part   = -0.1 * (xs / 2.4) ** 2

    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(t, th1, label="theta1", color="steelblue")
    axes[0].plot(t, th2, label="theta2", color="darkorange")
    axes[0].axhline(0,   color="green", ls="--", lw=1, label="upright")
    axes[0].axhline(180, color="red",   ls=":",  lw=1, label="hanging down")
    axes[0].axhline(-180,color="red",   ls=":",  lw=1)
    axes[0].set_ylabel("Angle (deg)")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_title("Double CartPole Swing-Up — trajectory from hanging-down")

    axes[1].plot(t, w1, label="w1", color="steelblue")
    axes[1].plot(t, w2, label="w2", color="darkorange")
    axes[1].axhline(0, color="gray", ls="--", lw=1)
    axes[1].set_ylabel("Angular vel (rad/s)")
    axes[1].legend(loc="upper right", fontsize=9)

    axes[2].plot(t, xs,  label="x",     color="black")
    axes[2].plot(t, xds, label="x_dot", color="gray")
    axes[2].axhline( 2.4, color="red",   ls=":",  lw=1)
    axes[2].axhline(-2.4, color="red",   ls=":",  lw=1)
    axes[2].axhline(0,    color="green", ls="--", lw=1)
    axes[2].set_ylabel("Cart x (m), x_dot (m/s)")
    axes[2].legend(loc="upper right", fontsize=9)

    axes[3].plot(t, E, color="purple")
    axes[3].axhline( 1.0, color="green", ls="--", lw=1, label="E_target")
    axes[3].axhline(-1.0, color="red",   ls=":",  lw=1, label="hanging rest")
    axes[3].set_ylabel("E_pole / E_target")
    axes[3].legend(loc="lower right", fontsize=9)

    axes[4].plot(t, rs,        color="purple", label="total")
    axes[4].plot(t, cos_part,  color="steelblue", lw=0.8, label="cos part")
    axes[4].plot(t, x_part,    color="black",     lw=0.8, label="x penalty")
    axes[4].axhline(0,   color="gray",  ls="--", lw=1)
    axes[4].axhline(1.0, color="green", ls="--", lw=1, label="max reward")
    axes[4].set_ylabel("Reward")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to {save_path.resolve()}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Double CartPole Swing-Up — CUDA Policy Iteration"
    )
    parser.add_argument("--render",    action="store_true",
                        help="Render evaluation episodes with pygame")
    parser.add_argument("--random",    type=int, nargs="?", const=3, default=None, metavar="N",
                        help="Run N random-policy episodes (physics check, no training)")
    parser.add_argument("--record",    type=Path, default=None, metavar="PATH",
                        help="Save evaluation video to PATH (.gif or .mp4)")
    parser.add_argument("--episodes",  type=int, default=3,
                        help="Number of evaluation episodes (default: 3)")
    parser.add_argument("--steps",     type=int, default=1000,
                        help="Max steps per episode (default: 1000)")
    parser.add_argument("--bins",      type=int, default=BINS_PER_DIM,
                        help=f"Bins per dimension (default: {BINS_PER_DIM}). "
                             "Memory: 15->~420MB, 20->~2.1GB")
    parser.add_argument("--seed",      type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-plot",   action="store_true",
                        help="Skip saving plots")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if a saved policy exists")
    parser.add_argument("--save-path", type=Path,
                        default=Path("results/double_cartpole_swingup_cuda_policy.npz"))
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
            pi = DoubleCartPoleSwingUpCuda.load(args.save_path)
        else:
            print("[*] Training new policy...")
            pi = train(args.save_path)

        evaluate(pi, n_episodes=args.episodes, render=args.render,
                 record_path=args.record, seed=args.seed, max_steps=args.steps)
        if not args.no_plot:
            plot_value_slice(pi)
            plot_trajectory(pi, seed=args.seed, max_steps=args.steps)
