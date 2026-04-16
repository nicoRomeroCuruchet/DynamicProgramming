# CUDA Policy Iteration — Continuous RL via Dynamic Programming

Solve continuous-state, continuous-action control problems with GPU-accelerated Policy Iteration.  
The implementation is based on the theoretical framework described in **[Continuous RL via Dynamic Programming](continuous_RL.pdf)**, which rigorously bridges the gap between continuous physics (Hamilton-Jacobi-Bellman equations) and discrete computation (Markov Decision Processes).

---

## Table of Contents

1. [Theoretical Background](#theoretical-background)
2. [Environments](#environments)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Running the Runners](#running-the-runners)
6. [CLI Reference](#cli-reference)

---

## Theoretical Background

### The Continuous Optimal Control Problem

The framework targets systems governed by first-order ODEs:

$$\dot{x}(t) = f(x(t), u(t)), \quad x(0) = x_0$$

where $x(t) \in \Omega \subset \mathbb{R}^d$ is the continuous state and $u(t) \in \mathcal{U}$ is the control input. The goal is to find a policy $\pi: \Omega \to \mathcal{U}$ that maximises the discounted cumulative reward:

$$V(x) = \sup_{u(\cdot)} \left\{ \int_0^{\infty} \gamma^t\, r(x(t), u(t))\, dt \right\}, \quad \gamma \in (0, 1)$$

### Hamilton-Jacobi-Bellman Equation

The exact optimality condition is the **HJB PDE**:

$$0 = V(x)\ln\gamma + \sup_{u \in \mathcal{U}} \left\{ \nabla V(x) \cdot f(x, u) + r(x, u) \right\}$$

This equation is analytically intractable for all but the simplest systems. The paper provides a rigorous discretisation scheme that converges to its solution.

### Bellman Principle & Discretisation

The **Dynamic Programming Principle** (Theorem 2.3) decomposes the value function:

$$V(x) = \sup_{u \in \mathcal{U}} \left\{ r(x, u)\,\tau + \gamma^\tau\, V\!\left(x + \tau f(x, u)\right) \right\} + \mathcal{O}(\tau^2)$$

The continuous state space $\Omega$ is discretised into a **simplicial mesh** $\Sigma^\delta$ with maximum diameter $\delta$. When the forward Euler step $\eta = x + \tau f(x, u)$ lands off-grid, the value is recovered via **barycentric interpolation** over the enclosing simplex:

$$V(\eta) \approx \sum_i p(\eta \mid \xi_i)\, V(\xi_i)$$

where the weights $p(\eta \mid \xi_i) \geq 0$ satisfy $\sum_i p = 1$ and $\sum_i p\, \xi_i = \eta$. This interpolation is exact (second-order) and requires no explicit error correction.

The resulting **barycentric dynamic programming operator** $F^\delta$ is a **contraction mapping** (Lemma 2.7):

$$\| F^\delta[V] - F^\delta[W] \|_\infty \leq \underbrace{\gamma^{\tau_{\min}}}_{\lambda < 1} \| V - W \|_\infty$$

By the Banach Fixed-Point Theorem, iterating $V_{k+1} = F^\delta[V_k]$ converges to a unique fixed point $V^\delta$ — the discrete approximation of the continuous value function — regardless of the initialisation.

### Policy Iteration

**Policy Iteration** alternates two steps until convergence:

| Step | Operation |
|---|---|
| **Policy Evaluation** | Iterate the Bellman expectation operator for the current policy $\pi$ until $\|V_{k+1} - V_k\|_\infty < \theta$ |
| **Policy Improvement** | Greedily update $\pi(s) \leftarrow \arg\max_a \sum_{s'} p(s'\|s,a)[r + \gamma V(s')]$ |

Theorem 3.1 guarantees convergence to the **globally optimal policy** in a finite number of iterations.

### CUDA Parallelisation

Each GPU thread independently evaluates the Bellman supremum at one grid node:

```
for each node ξᵢ  (in parallel, one CUDA thread per node)
    for each action u ∈ U
        η  ← ξᵢ + τ · f(ξᵢ, u)          // forward Euler
        V* ← barycentric_interp(η, V)     // 2ᵈ-corner lookup
        Q  ← r(ξᵢ, u) + γ^τ · V*
    π(ξᵢ) ← argmax Q
    V(ξᵢ)  ← max Q
```

No inter-thread communication is needed during evaluation — the entire value update is embarrassingly parallel.

---

## Environments

All runners share the same CUDA Policy Iteration core. Each environment injects its own dynamics as a raw CUDA C device function via `_dynamics_cuda_src()`, which is compiled at runtime with **NVRTC**.

### CartPole

Classic inverted pendulum. A force is applied to a cart to keep a pole balanced upright.

| State | Bounds |
|---|---|
| Cart position $x$ | $[-2.5, 2.5]$ m |
| Cart velocity $\dot{x}$ | $[-5, 5]$ m/s |
| Pole angle $\theta$ | $[-0.25, 0.25]$ rad |
| Pole angular velocity $\dot{\theta}$ | $[-5, 5]$ rad/s |

**Actions:** $\{-10, +10\}$ N &nbsp;|&nbsp; **Grid:** $30^4$ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration4D`

![CartPole](gifs/cartpole.gif)

---

### Pendulum

Swing up and stabilise a free pendulum at the upright position.

| State | Bounds |
|---|---|
| Angle $\theta$ | $[-\pi, \pi]$ rad |
| Angular velocity $\dot{\theta}$ | $[-8, 8]$ rad/s |

**Actions:** 21 torque values in $[-2, 2]$ N·m &nbsp;|&nbsp; **Grid:** $200^2$ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration2D`

![Pendulum](gifs/pendulum.gif)

---

### Mountain Car

Drive an underpowered car out of a valley. Requires learning to build momentum by oscillating.

| State | Bounds |
|---|---|
| Position | $[-1.2, 0.6]$ |
| Velocity | $[-0.07, 0.07]$ |

**Actions:** $\{-1, 0, +1\}$ &nbsp;|&nbsp; **Grid:** $200^2$ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration2D`

![Mountain Car](gifs/mountain_car.gif)

---

### Continuous Mountain Car

Same as Mountain Car but with a continuous action space.

| State | Bounds |
|---|---|
| Position | $[-1.2, 0.6]$ |
| Velocity | $[-0.07, 0.07]$ |

**Actions:** 21 force values in $[-1, 1]$ &nbsp;|&nbsp; **Grid:** $200^2$ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration2D`

![Continuous Mountain Car](gifs/continuous_mountain_car.gif)

---

### Double CartPole

Two poles of different lengths balanced simultaneously on a single cart. 6-dimensional state space.

| State | Bounds |
|---|---|
| Cart position $x$ | $[-2.4, 2.4]$ m |
| Cart velocity $\dot{x}$ | $[-3, 3]$ m/s |
| Pole 1 angle $\theta_1$ | $[-0.3, 0.3]$ rad |
| Pole 1 angular velocity $\dot{\theta}_1$ | $[-3, 3]$ rad/s |
| Pole 2 angle $\theta_2$ | $[-0.3, 0.3]$ rad |
| Pole 2 angular velocity $\dot{\theta}_2$ | $[-3, 3]$ rad/s |

**Actions:** $\{-10, 0, +10\}$ N &nbsp;|&nbsp; **Grid:** $12^6$ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration6D`

> **Note:** Memory scales as $B^6$. Recommended: `--bins 12` (~90 MB), `--bins 15` (~420 MB), `--bins 20` (~2.1 GB).

![Double CartPole](gifs/double_cartpole.gif)

---

### Overhead Crane (Anti-Sway)

A trolley moves along a fixed rail carrying a suspended load. Goal: transport the load from one end of the rail to the other while minimising swing. Lagrangian dynamics with a 2×2 mass matrix solved analytically at each step.

$$H = \begin{bmatrix} M + m & mL\cos\theta \\ mL\cos\theta & mL^2 \end{bmatrix}, \qquad \det(H) = mL^2(M + m\sin^2\theta) > 0$$

| State | Bounds |
|---|---|
| Trolley position $x$ | $[-3, 3]$ m |
| Trolley velocity $\dot{x}$ | $[-4, 4]$ m/s |
| Rope angle $\theta$ | $[-\pi/2 \cdot 1.1,\; \pi/2 \cdot 1.1]$ rad |
| Rope angular velocity $\dot{\theta}$ | $[-4, 4]$ rad/s |

**Physical parameters:** $M = 1$ kg (trolley), $m = 5$ kg (load), $L = 1.5$ m (rope), $g = 9.81$ m/s²

**Actions:** $\{-30, -15, 0, +15, +30\}$ N &nbsp;|&nbsp; **Grid:** $30^4$ nodes &nbsp;|&nbsp; **Base class:** `CudaPolicyIteration4D`

**Reward:** $1 - 0.15\,x_n^2 - 0.15\,\dot{x}_n^2 - 0.45\,\theta_n^2 - 0.25\,\dot{\theta}_n^2$, with each term normalised by its respective grid bound.

![Overhead Crane](gifs/overhead_crane.gif)

---

## Project Structure

```
DynamicProgramming/
├── src/
│   └── cuda_policy_iteration.py   # Core engine: CudaPolicyIteration2D/4D/6D + CudaPIConfig
├── runners/
│   ├── cartpole_cuda.py
│   ├── pendulum_cuda.py
│   ├── mountain_car_cuda.py
│   ├── continuous_mountain_car_cuda.py
│   ├── double_cartpole_cuda.py
│   └── overhead_crane_cuda.py
├── utils/
│   └── barycentric.py             # Barycentric interpolation for policy inference
├── gifs/                          # Pre-recorded environment demos
├── results/                       # Saved policies (.npz) and plots
├── requirements.txt
└── continuous_RL.pdf              # Theoretical foundation
```

Each runner subclasses the appropriate base class and implements a single method:

```python
def _dynamics_cuda_src(self) -> str:
    """Return a raw CUDA C string containing the step_dynamics() device function."""
```

The kernel is compiled at runtime via **NVRTC** — no pre-compilation step needed.

---

## Setup

### 1. Prerequisites

- **NVIDIA GPU** with CUDA Toolkit installed
- **Python 3.10+**
- `nvidia-smi` available in PATH

Check your CUDA version:

```bash
nvidia-smi | grep "CUDA Version"
# or
nvcc --version
```

### 2. Create a Virtual Environment

```bash
python3 -m venv env
source env/bin/activate        # Linux / macOS
# env\Scripts\activate         # Windows
```

### 3. Install CuPy — Match Your CUDA Version

> **Critical:** CuPy must match the CUDA Toolkit installed on your machine. Installing the wrong version will fail silently or raise `cupy.cuda.compiler.CompileException`.

| CUDA Version | Package |
|---|---|
| 11.x | `cupy-cuda11x` |
| 12.x | `cupy-cuda12x` |
| 13.x | `cupy-cuda13x` |

Edit `requirements.txt` to match your version, then install:

```bash
# Example for CUDA 12.x (default in requirements.txt)
pip install cupy-cuda12x

# Example for CUDA 11.x
pip install cupy-cuda11x
```

Verify CuPy can see your GPU:

```bash
python3 -c "import cupy; print(cupy.cuda.runtime.getDeviceCount(), 'GPU(s) detected')"
```

### 4. Install All Dependencies

```bash
pip install -r requirements.txt
```

For MP4 video recording (optional):

```bash
pip install "imageio[ffmpeg]"
```

### 5. Final Check

```bash
python3 -c "
import cupy, numba, numpy, gymnasium, matplotlib
print('cupy   :', cupy.__version__)
print('numba  :', numba.__version__)
print('numpy  :', numpy.__version__)
print('gym    :', gymnasium.__version__)
print('All OK')
"
```

---

## Running the Runners

All runners follow the same workflow:

```
cd DynamicProgramming/

# First run: trains and saves policy, then evaluates
python3 runners/<runner>.py

# Subsequent runs: loads saved policy, skips training
python3 runners/<runner>.py

# Force retraining
python3 runners/<runner>.py --retrain

# Render live + record to GIF
python3 runners/<runner>.py --render --record gifs/<runner>.gif --episodes 3

# Quick random baseline (no training needed)
python3 runners/<runner>.py --random 5
```

### Generate All GIFs

```bash
python3 runners/cartpole_cuda.py                --record gifs/cartpole.gif                --episodes 3 --no-plot
python3 runners/pendulum_cuda.py                --record gifs/pendulum.gif                --episodes 3 --no-plot
python3 runners/mountain_car_cuda.py            --record gifs/mountain_car.gif            --episodes 3 --no-plot
python3 runners/continuous_mountain_car_cuda.py --record gifs/continuous_mountain_car.gif --episodes 3 --no-plot
python3 runners/double_cartpole_cuda.py         --record gifs/double_cartpole.gif         --episodes 3 --no-plot
python3 runners/overhead_crane_cuda.py          --record gifs/overhead_crane.gif          --episodes 3 --no-plot
```

### Overhead Crane: Custom Target

```bash
# Transport from x=+2.5 m to x=-2.5 m (default)
python3 runners/overhead_crane_cuda.py --retrain --target-x -2.5 --start-x 2.5

# Transport to centre
python3 runners/overhead_crane_cuda.py --retrain --target-x 0.0 --start-x 2.5
```

---

## CLI Reference

All runners share a common set of arguments:

| Argument | Default | Description |
|---|---|---|
| `--render` | off | Open a live window during evaluation |
| `--record PATH` | off | Save video to `.gif` or `.mp4` (MP4 needs `imageio[ffmpeg]`) |
| `--random [N]` | off | Run N episodes with a random policy as baseline (default N=5) |
| `--episodes N` | 5 | Number of evaluation episodes |
| `--bins N` | varies | Bins per state dimension — trades memory for resolution |
| `--seed N` | 42 | Random seed for episode resets |
| `--no-plot` | off | Skip saving value function / policy slice plots |
| `--retrain` | off | Force retraining even if a saved policy exists |
| `--save-path PATH` | `results/…` | Where to save / load the policy `.npz` |

**Overhead Crane only:**

| Argument | Default | Description |
|---|---|---|
| `--target-x X` | `-2.5` | Target trolley position (m) — baked into CUDA at compile time |
| `--start-x X` | `2.5` | Initial trolley position for evaluation (m) |

### Memory Guide for `--bins`

| Problem | Dims | `--bins 20` | `--bins 30` | `--bins 40` |
|---|---|---|---|---|
| Pendulum, Mountain Car | 2D | < 1 MB | < 1 MB | < 1 MB |
| CartPole, Overhead Crane | 4D | ~25 MB | ~130 MB | ~408 MB |
| Double CartPole | 6D | ~2.1 GB | ~24 GB* | — |

> *6D grids scale very steeply. For Double CartPole, `--bins 12` (≈90 MB) or `--bins 15` (≈420 MB) are practical choices.

---

## Extending with a New Environment

1. Subclass `CudaPolicyIteration2D`, `4D`, or `6D` from `src/cuda_policy_iteration.py`
2. Implement `_dynamics_cuda_src()` returning a CUDA C string with a `step_dynamics()` device function
3. Define `BINS_SPACE`, `ACTION_SPACE`, and a `train()` / `evaluate()` wrapper
4. Optionally override `_terminal_fn()` to mark absorbing states and `_allocate_tensors_and_compile()` to set their values

```python
class MyEnvCuda(CudaPolicyIteration4D):
    def _dynamics_cuda_src(self) -> str:
        return r'''
        __device__ void step_dynamics(
            float s0, float s1, float s2, float s3, float action,
            float* ns0, float* ns1, float* ns2, float* ns3,
            float* reward, bool* terminated
        ) {
            // your Euler-step dynamics here
        }
        '''
```

The CUDA string is compiled once at the start of `run()` via NVRTC — no Makefile or `.cu` files needed.
