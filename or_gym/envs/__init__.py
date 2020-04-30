from gym.envs.registration import register

register(id='Knapsack-v0',
    entry_point='or_gym.envs.classic_or.knapsack:KnapsackEnv'
)

register(id='Knapsack-v1',
	entry_point='or_gym.envs.classic_or.knapsack:BoundedKnapsackEnv'
)

register(id='Knapsack-v2',
	entry_point='or_gym.envs.classic_or.knapsack:OnlineKnapsackEnv'
)

register(id='BinPacking-v0',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingEnv'
)

register(id='VMPacking-v0',
	entry_point='or_gym.envs.classic_or.vmpacking:VMPackingEnv'
)

register(id='VMPacking-v1',
	entry_point='or_gym.envs.classic_or.vmpacking:TempVMPackingEnv'
)

register(id='VehicleRouting-v1',
	entry_point='or_gym.envs.classic_or.vehicle_routing:VehicleRoutingEnv'
)

register(id='NewsVendor-v1',
	entry_point='or_gym.envs.classic_or.newsvendor:NewsVendorBacklogEnv'
)

register(id='NewsVendor-v2',
	entry_point='or_gym.envs.classic_or.newsvendor:NewsVendorLostSalesEnv'
)

register(id='PortfolioOpt-v0',
	entry_point='or_gym.envs.finance.portfolio_opt:PortfolioOptEnv'
)