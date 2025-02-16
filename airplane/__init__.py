from gymnasium.envs.registration import register

    
register(
    id="ReducedSymmetricGliderPullout-v0",
    entry_point="airplane.reduced_symmetric_glider_pullout:ReducedSymmetricGliderPullout",
    max_episode_steps=100,
)

    
register(
    id="ReducedBankedGliderPullout-v0",
    entry_point="airplane.reduced_banked_glider_pullout:ReducedBankedGliderPullout",
    max_episode_steps=100,
)