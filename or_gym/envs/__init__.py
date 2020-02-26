from gym.envs.registration import register

register(id='Knapsack-v0',
    entry_point='or_gym.envs.classic_or:KnapsackEnv'
)
