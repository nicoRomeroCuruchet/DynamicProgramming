import pickle
import numpy as np
import matplotlib.pyplot as plt
from PolicyIteration import PolicyIteration
#from classic_control.cartpole import CartPoleEnv
from classic_control.continuous_mountain_car import Continuous_MountainCarEnv


from classic_control.continuous_mountain_car import Continuous_MountainCarEnv

env=Continuous_MountainCarEnv()

bins_space = {
    "x_space":     np.linspace(env.min_position, env.max_position, 100,      dtype=np.float32),    # position space    (0)
    "x_dot_space": np.linspace(-abs(env.max_speed), abs(env.max_speed), 100, dtype=np.float32),    # velocity space    (1)
}

pi = PolicyIteration(
    env=env, 
    bins_space=bins_space,
    action_space=np.linspace(-1.0, +1.0,9, dtype=np.float32),
    gamma=0.99,
    theta=1e-3,
)

pi.run()

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
# decrease ticks
ax.set_xticks(np.linspace(env.min_position, env.max_position, 4))
ax.set_yticks(np.linspace(-abs(env.max_speed), abs(env.max_speed), 4))
# Add labels
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('Normalized Value Function')
# Add color bar to represent the value range
#fig.colorbar(surf)
plt.show()