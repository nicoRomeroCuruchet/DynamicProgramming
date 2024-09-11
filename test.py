import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import numpy as np
from PolicyIteration import PolicyIteration
from utils.utils import plot_2D_value_function, plot_3D_value_function, test_enviroment

from classic_control.cartpole import CartPoleEnv
from classic_control.continuous_mountain_car import Continuous_MountainCarEnv

# 00:43<05:47
env=Continuous_MountainCarEnv()

bins_space = {
    "x_space":     np.linspace(env.min_position, env.max_position, 50,      dtype=np.float32),    # position space    (0)
    "x_dot_space": np.linspace(-abs(env.max_speed), abs(env.max_speed), 50, dtype=np.float32),    # velocity space    (1)
}

pi = PolicyIteration(
    env=env, 
    bins_space=bins_space,
    action_space=np.linspace(-1.0, +1.0,15, dtype=np.float32),
    gamma=0.99,
    theta=1e-3,
)
pi.run()









env = Continuous_MountainCarEnv()


with open(env.__class__.__name__ + ".pkl", "rb") as f:
    pi: PolicyIteration = pickle.load(f)


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
