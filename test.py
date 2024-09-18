import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import numpy as np
from PolicyIteration import PolicyIteration
from utils.utils import plot_2D_value_function, plot_3D_value_function, test_enviroment

from classic_control.cartpole import CartPoleEnv
from classic_control.continuous_mountain_car import Continuous_MountainCarEnv

env = Continuous_MountainCarEnv()


with open(env.__class__.__name__ + ".pkl", "rb") as f:
    pi: PolicyIteration = pickle.load(f)


# Assuming 'pi' contains the state space and value function as provided
vf = pi.value_function
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

# Show the plot
plt.show()
1/0

points = pi.states_space
# Assuming points is a 2D array where each row is a point [position, velocity]
positions = points[:, 0]  # x-axis (position)
velocities = points[:, 1]  # y-axis (velocity)
values = pi.value_function  # z-axis (value function)

# normalize values
# normalize the value function
min_value = np.min(values)
max_value = np.max(values)
values = (
    (values - min_value) / (max_value - min_value)
    if max_value > min_value
    else np.zeros_like(values)
)

# Determine the unique grid sizes
x_unique = np.unique(positions)
y_unique = np.unique(velocities)

# Reshape the position, velocity, and value arrays into 2D grids
X, Y = np.meshgrid(x_unique, y_unique)

Z = values.reshape(X.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the surface
surf = ax.plot_surface(-X, Y, Z, cmap="turbo_r", edgecolor="w")
ax.set_xticks(np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=3))
ax.set_yticks(np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), num=4))

# Label the axes
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
ax.set_zlabel("Value Function")

#Add color bar to represent the value range
#fig.colorbar(surf)

# Show the plot
plt.show()

# doing a 3d quiver plot
# x,y is the state (position, velocity)
# z is the value function
# u,v is the observed state after taking the action
# w is 0

# save the transition with the greedy action
obs_max = np.zeros_like(pi.states_space)  # 625x2
reward_max = np.zeros_like(pi.states_space[:, 0])


# reflect states along x axis
states = np.copy(pi.states_space)

for action in pi.action_space:
    obs, reward, _, _, _ = pi.step(states, action)
    obs_max = np.where(reward[:, None] > reward_max[:, None], obs, obs_max)


u = states[:, 0] - obs_max[:, 0]
v = states[:, 1] - obs_max[:, 1]
w = np.zeros_like(u)

"""ax.quiver(
    -X.flatten(),
    Y.flatten(),
    Z.flatten(),
    u,
    v,
    w,
    color="r",
    length=0.1,  # Why scale?
    normalize=False,
)"""
# Show the plot
plt.show()
