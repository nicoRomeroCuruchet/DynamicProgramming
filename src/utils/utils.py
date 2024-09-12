import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

def plot_2D_value_function(data: dict, 
                           normalize: bool = True, 
                           cmap:str='OrRd_r',
                           show:bool=True,
                           number:int=None,
                           path:str='')->None:
    """
    Plots a 2D value function in some color scale.

    Parameters:
    - data (dict): A dictionary containing the value function data. The keys represent the position and velocity, and the values represent the meaning.
    - normalize (bool): Whether to normalize the value function. Defaults to True.
    - cmap (str): The colormap to use for the contour plot. Defaults to "OrRd_r".

    Returns:
    None
    """
    # Extract position, velocity, and meaning from the dictionary
    positions = np.array([key[0] for key in data.keys()])
    velocities = np.array([key[1] for key in data.keys()])
    meanings = np.array(list(data.values()))
    max = np.max(meanings) if normalize else 1.0

    # Create grid data
    pos_grid, vel_grid = np.meshgrid(np.unique(positions), np.unique(velocities))

    # Create a grid for the meanings
    mean_grid = np.zeros_like(pos_grid, dtype=float)

    for i in range(pos_grid.shape[0]):
        for j in range(pos_grid.shape[1]):
            pos = pos_grid[i, j]
            vel = vel_grid[i, j]
            mean_grid[i, j] = data.get((pos, vel), np.nan) / max # Use np.nan for missing values

    # Create a filled contour plot
    fig, ax = plt.subplots()

    # Create the contour plot with grayscale color map
    contour = ax.contourf(pos_grid, vel_grid, mean_grid, cmap=cmap)

    # Add contour lines
    contour_lines = ax.contour(pos_grid, 
                            vel_grid, 
                            mean_grid, 
                            colors='k',
                            linewidths=0.45, 
                            linestyles='dashed')

    # Add labels to the contour lines
    ax.clabel(contour_lines, inline=True, fontsize=8)
    # Set labels
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    # Mark the target position
    target_position = 0.45
    target_velocity = 0.0
    ax.plot(target_position, target_velocity, 'kx')  # 'rx' means red color and x marker
    ax.text(target_position, target_velocity, ' Target', color='k', fontsize=10, ha='left', va='bottom')
    # Add a color bar which maps values to colors
    cbar = fig.colorbar(contour)
    cbar.set_label('Normalize value function')
    if number is not None:
        fig.text(0.0001, 1, "Iteration "+str(number), transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # save the plot in the parent directory
    plt.savefig(path)
    # Show the plot
    if show: plt.show()
    plt.close()
    
def plot_3D_value_function(vf: np.array,
                           points: np.array,
                           normalize: bool = True, 
                           cmap:str='OrRd_r',
                           show:bool=True,
                           number:int=None, 
                           path:str='')->None:
    """
    Plots a 3D surface plot of a value function.

    Parameters:
    - vf (dict): A dictionary representing the value function. The keys are tuples of (position, velocity),
                 and the values are the corresponding meanings.
    - normalize (bool): Whether to normalize the value function. If True, the values will be normalized between 0 and 1.

    Returns:
    None
    """

    # Assuming points is a 2D array where each row is a point [position, velocity]
    positions  = points[:, 0]       # x-axis (position)
    velocities = points[:, 1]       # y-axis (velocity)
    values     = vf                 # z-axis (value function)
    
    # normalize the value function
    if normalize:
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
    surf = ax.plot_surface(X, -Y, Z, cmap='turbo_r', edgecolor='w')
    ax.set_xticks(np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=3))
    ax.set_yticks(np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), num=4))

    # Label the axes
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value Function')

    plt.savefig(path)
    # Show the plot
    if show: plt.show()
    plt.close()


def get_barycentric_coordinates(op:np.array, point:np.array)->tuple:

    simplex_index = op.triangulation.find_simplex(point)
    if simplex_index != -1:  # -1 indicates that the point is outside the convex hull
        points_indexes = op.triangulation.simplices[simplex_index]
    else:
        # raise an error
        raise ValueError(f"The point {point} is outside the convex hull.")
    
    simplex = op.states_space[points_indexes]
    simplex = np.array(simplex, dtype=np.float32).reshape(op.num_simplex_points, op.space_dim)

    A = np.vstack([simplex.T, np.ones(len(simplex))])
    b = np.hstack([point, [1]]).reshape(op.space_dim+1,)       
    try:
        inv_A = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        #penrose-Moore pseudo inverse and log
        inv_A = np.linalg.pinv(A)
        print(f"Error: {e}")
    
    # get barycentric coordinates: lambdas = A^-1 * b
    lambdas = np.array(inv_A@b.T,dtype=np.float32).reshape(1,op.num_simplex_points)
    # Check if the point is inside the simplex
    if np.any(lambdas < -1.0e-2) and abs(np.sum(lambdas) - 1.0) > 1.0e-2:
        raise ValueError(f"The point {point} is outside the convex hull.")
    
    return lambdas, (simplex, points_indexes)
    
def get_optimal_action(state:np.array, optimal_policy:np.array):
    """
    Aproximate the optimal action for a given state using the provided optimal policy
    with barycentric interpolation.

    Parameters:
    state (np.array): The state for which to determine the optimal action.
    optimal_policy (PolicyIteration): The optimal policy used to determine the action.

    Returns:
    action: The optimal action for the given state.
    """    
    lambdas, simmplex_info  = get_barycentric_coordinates(optimal_policy, state)
    simplex, points_indexes = simmplex_info
    actions = optimal_policy.action_space
    probabilities = np.zeros(len(actions), dtype=np.float32)

    if np.linalg.norm(np.array(lambdas, dtype=np.float32).dot(simplex) - state) > 1e-2 :
        raise ValueError("The state is not in the linear combination by the vertices of the simplex")

    # reshape the lambdas to be a column vector to use enumerate
    lambdas = np.array(lambdas).reshape(-1, 1)
    for i, l in enumerate(lambdas):
        for j, action in enumerate(actions):
            probabilities[j] += l * float(optimal_policy.policy[points_indexes[i]][j])
    
    if abs(np.sum(probabilities)-1) > 1e-2:
        raise ValueError(f"The probabilities do not sum to 1, sum is {np.sum(probabilities)}")
    
    argmax = lambda x: max(enumerate(x), key=lambda x: x[1])[0]
    action = actions[argmax(probabilities)]
    return action

def test_enviroment(task: gym.Env, 
                    pi, 
                    num_episodes:  int  = 10000, 
                    episode_lengh: int  = 1000,
                    option_reset:  dict = None):
    """
    Test the environment using the given policy iteration algorithm.

    Parameters:
    - task (gym.Env): The environment to test.
    - pi (PolicyIteration): The policy iteration algorithm.
    - num_episodes (int): The number of episodes to run. Default is 10000.
    - episode_lengh (int): The maximum length of each episode. Default is 1000.
    """

    for episode in range(0, num_episodes):
        total_reward = 0
        observation, _ = task.reset(options=option_reset)
        for timestep in range(1, episode_lengh):
            action = get_optimal_action(observation, pi)
            observation, reward, terminated, _, _ = task.step(action)
            total_reward += reward
            if terminated:
                print(f"Episode {episode} finished after {timestep} timesteps")
                print(f"Total reward: {total_reward}")
                break