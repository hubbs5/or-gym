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

register(id='InvManagement-v0',
	entry_point='or_gym.envs.supply_chain.inventory_management:InvManagementBacklogEnv'
)

register(id='InvManagement-v1',
	entry_point='or_gym.envs.supply_chain.inventory_management:InvManagementLostSalesEnv'
)

register(id='PortfolioOpt-v0',
	entry_point='or_gym.envs.finance.portfolio_opt:PortfolioOptEnv'
)

register(id='SCSched-v0',
	entry_point='or_gym.envs.supply_chain.scheduling:DiscreteSchedEnv'
)

register(id='SCSched-v1',
	entry_point='or_gym.envs.supply_chain.scheduling:MaskedDiscreteSchedEnv'
)

register(id='SCSched-v2',
	entry_point='or_gym.envs.supply_chain.scheduling:ContSchedEnv'
)

register(id='SCSched-v3',
	entry_point='or_gym.envs.supply_chain.scheduling:MaskedContSchedEnv'
)