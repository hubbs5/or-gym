from gym.envs.registration import register

# Knapsack Environments
register(id='Knapsack-v0',
    entry_point='or_gym.envs.classic_or.knapsack:KnapsackEnv'
)

register(id='Knapsack-v1',
	entry_point='or_gym.envs.classic_or.knapsack:BinaryKnapsackEnv'
)

register(id='Knapsack-v2',
	entry_point='or_gym.envs.classic_or.knapsack:BoundedKnapsackEnv'
)

register(id='Knapsack-v3',
	entry_point='or_gym.envs.classic_or.knapsack:OnlineKnapsackEnv'
)

# Bin Packing Environments
register(id='BinPacking-v0',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingEnv'
)

register(id='BinPacking-v1',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingLW1'
)

register(id='BinPacking-v2',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingPP0'
)

register(id='BinPacking-v3',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingPP1'
)

register(id='BinPacking-v4',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingBW0'
)

register(id='BinPacking-v5',
	entry_point='or_gym.envs.classic_or.binpacking:BinPackingBW1'
)

# Newsvendor Envs
register(id='Newsvendor-v0',
	entry_point='or_gym.envs.classic_or.newsvendor:NewsvendorEnv'
)

# Virtual Machine Packing Envs
register(id='VMPacking-v0',
	entry_point='or_gym.envs.classic_or.vmpacking:VMPackingEnv'
)

register(id='VMPacking-v1',
	entry_point='or_gym.envs.classic_or.vmpacking:TempVMPackingEnv'
)

# Vehicle Routing Envs
register(id='VehicleRouting-v0',
	entry_point='or_gym.envs.classic_or.vehicle_routing:VehicleRoutingEnv'
)

# TSP
register(id='TSP-v0',
	entry_point='or_gym.envs.classic_or.tsp:TSPEnv'
)

register(id='TSP-v1',
	entry_point='or_gym.envs.classic_or.tsp:TSPDistCost'
)

# Inventory Management Envs
register(id='InvManagement-v0',
	entry_point='or_gym.envs.supply_chain.inventory_management:InvManagementBacklogEnv'
)

register(id='InvManagement-v1',
	entry_point='or_gym.envs.supply_chain.inventory_management:InvManagementLostSalesEnv'
)

register(id='NetworkManagement-v0',
	entry_point='or_gym.envs.supply_chain.network_management:NetInvMgmtBacklogEnv'
)

register(id='NetworkManagement-v1',
	entry_point='or_gym.envs.supply_chain.network_management:NetInvMgmtLostSalesEnv'
)

# Asset Allocation Envs
register(id='PortfolioOpt-v0',
	entry_point='or_gym.envs.finance.portfolio_opt:PortfolioOptEnv'
)
