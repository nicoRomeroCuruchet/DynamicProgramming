import pickle
import numpy as np
import matplotlib.pyplot as plt
from PolicyIteration import PolicyIteration
from classic_control.cartpole import CartPoleEnv
from classic_control.continuous_mountain_car import Continuous_MountainCarEnv


env = CartPoleEnv(sutton_barto_reward=True)
# position thresholds:
x_lim         = 2.4
theta_lim     = 0.418 
# velocity thresholds:
x_dot_lim     = 3.1
theta_dot_lim = 3.1

bins_space = {
    "x_space"         : np.linspace(-x_lim, x_lim, 15,  dtype=np.float32),                     # position space          (0)
    "x_dot_space"     : np.linspace(-x_dot_lim, x_dot_lim, 15,  dtype=np.float32),             # velocity space          (1)
    "theta_space"     : np.linspace(-theta_lim, theta_lim, 15, dtype=np.float32),              # angle space             (2)
    "theta_dot_space" : np.linspace(-theta_dot_lim, theta_dot_lim, 15, dtype=np.float32),      # angular velocity space  (3)
}

pi = PolicyIteration(
    env=env, 
    bins_space=bins_space,
    action_space=np.array([0, 1], dtype=np.int32),
    gamma=0.99,
    theta=1e-3
)

pi.run()

with open(env.__class__.__name__ + ".pkl", "rb") as f:
    pi: PolicyIteration = pickle.load(f)


# Assuming 'pi' contains the state space and value function as provided
vf = pi.value_function.get()
# Normalize the value function data
normalized_vf = (vf - vf.min()) / (vf.max() - vf.min())
# Flight path angle and airspeed normalization data
flight_path_angle, airspeed_norm = pi.states_space[:, 0], pi.states_space[:, 1]
#flight_path_angle = np.degrees(flight_path_angle)  # Convert to degrees
# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Use plot_trisurf for unstructured triangular surface plot
surf = ax.plot_trisurf(flight_path_angle, airspeed_norm, normalized_vf, cmap='turbo_r', edgecolor='white', linewidth=0.2)
#flight_path_angle_grid, airspeed_norm_grid = np.meshgrid(np.unique(flight_path_angle), np.unique(airspeed_norm))
#normalized_vf_grid = normalized_vf.reshape(flight_path_angle_grid.shape)

#surf = ax.plot_surface(flight_path_angle_grid, airspeed_norm_grid, normalized_vf_grid, cmap='turbo_r')

# Add labels
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('Normalized Value Function')
# Add color bar to represent the value range
#fig.colorbar(surf)
plt.show()