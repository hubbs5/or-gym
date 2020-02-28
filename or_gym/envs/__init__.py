from gym.envs.registration import register

register(id='Knapsack-v0',
    entry_point='or_gym.envs.classic_or:KnapsackEnv'
)

register(id='Knapsack-v1',
	entry_point='or_gym.envs.classic_or:BoundedKnapsackEnv'
)

register(id='Knapsack-v2',
	entry_point='or_gym.envs.classic_or:OnlineKnapsackEnv'
)