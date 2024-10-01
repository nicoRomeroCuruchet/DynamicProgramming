from gymnasium.envs.registration import register

# Register the first custom environment
register(
    id='ReducedSymmetricGliderPullout-v0',
    entry_point='airplane.reduced_symmetric_glider_pullout:ReducedSymmetricGliderPullout', 
    max_episode_steps=100, 
)