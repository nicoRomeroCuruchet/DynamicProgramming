import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
    
def plot_3D_value_function(vf: np.array,
                           points: np.array,
                           normalize: bool = True, 
                           cmap:str='turbo_r',
                           show:bool=False,
                           path:str='')->None:
    
    """ Plots a 3D value function in some color scale."""
    # Assuming points is a 2D array where each row is a point [position, velocity]
    X  = points[:, 0]  # x-axis
    # transformb X from rad to deg
    X = np.rad2deg(X)
    Y  = points[:, 1]  # y-axis 
    vf = vf            # z-axis (value function)
    vf_to_plot = (vf - vf.min()) / (vf.max() - vf.min()) if normalize else vf
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Use plot_trisurf for unstructured triangular surface plot
    surf = ax.plot_trisurf(X, Y, vf_to_plot, cmap=cmap, edgecolor='white', linewidth=0.2)
    # Add title
    ax.set_title('Reduced Symmetric Glider Value Function ', pad=20)
    # Add labels
    ax.set_xlabel('Flight Path Angle (γ) [deg]', labelpad=10)
    ax.set_ylabel('V/Vs', labelpad=10)
    ax.set_zlabel('Normalized Value Function', labelpad=10)
    ax.set_xticks(np.linspace(min(X), max(X), 8))  # 5 ticks on the x-axis
    ax.set_yticks(np.linspace(min(Y), max(Y), 8))  # 5 ticks on the y-axis
    # Add color bar to represent the value range
    if path is not None: plt.savefig(path)
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
    lambdas, simmplex_info  = get_barycentric_coordinates(optimal_policy,state)
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
            observation, reward, terminated, _, _ = task.step_to_render(action)
            total_reward += reward
            if terminated or timestep == episode_lengh-1:
                print(f"Episode {episode} finished after {timestep} timesteps")
                print(f"Total reward: {total_reward}")
                break