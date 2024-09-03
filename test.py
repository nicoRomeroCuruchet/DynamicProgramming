import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pickle
import numpy as np
from PolicyIteration import PolicyIteration
from utils.utils import plot_2D_value_function,\
                        plot_3D_value_function,\
                        test_enviroment

from classic_control.cartpole import CartPoleEnv
from classic_control.continuous_mountain_car import Continuous_MountainCarEnv


env = Continuous_MountainCarEnv()


with open(env.__class__.__name__ + ".pkl", "rb") as f:
    pi: PolicyIteration = pickle.load(f)




points = pi.states_space
# Assuming points is a 2D array where each row is a point [position, velocity]
positions = points[:, 0]                # x-axis (position)
velocities = points[:, 1]               # y-axis (velocity)
values = pi.value_function              # z-axis (value function)

#normalize values
# normalize the value function
min_value = np.min(values)
max_value = np.max(values)
values = (values - min_value) / (max_value - min_value) if max_value > min_value else np.zeros_like(values)

# Determine the unique grid sizes
x_unique = np.unique(positions)
y_unique = np.unique(velocities)

# Reshape the position, velocity, and value arrays into 2D grids
X, Y = np.meshgrid(x_unique, y_unique)

Z = values.reshape(X.shape)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(-X, Y, Z, cmap='turbo_r', edgecolor='w')
ax.set_xticks(np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=3))
ax.set_yticks(np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), num=4))

# Label the axes
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_zlabel('Value Function')

# Show the plot
plt.show()