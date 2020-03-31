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

register(id='BinPacking-v0',
	entry_point='or_gym.envs.classic_or:BinPackingEnv'
)

register(id='VMPacking-v0',
	entry_point='or_gym.envs.classic_or:VMPackingEnv'
)

register(id='VMPacking-v1',
	entry_point='or_gym.envs.classic_or:TempVMPackingEnv'
)

register(id='NewsVendor-v1',
	entry_point='or_gym.envs.classic_or:MultiLevelNewsVendorEnv'
)