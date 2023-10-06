from gymnasium.envs.registration import register
from gymnasium.wrappers.compatibility import EnvCompatibility
from gymnasium.wrappers.step_api_compatibility import StepAPICompatibility

# Knapsack Environments
register(
    id="Knapsack-v0",
    entry_point="or_gym.envs.classic_or.knapsack:KnapsackEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

register(
    id="Knapsack-v1",
    entry_point="or_gym.envs.classic_or.knapsack:BinaryKnapsackEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

register(
    id="Knapsack-v2",
    entry_point="or_gym.envs.classic_or.knapsack:BoundedKnapsackEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

register(
    id="Knapsack-v3",
    entry_point="or_gym.envs.classic_or.knapsack:OnlineKnapsackEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

# Bin Packing Environments
register(
    id="BinPacking-v0", entry_point="or_gym.envs.classic_or.binpacking:BinPackingEnv", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

register(
    id="BinPacking-v1", entry_point="or_gym.envs.classic_or.binpacking:BinPackingLW1", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

register(
    id="BinPacking-v2", entry_point="or_gym.envs.classic_or.binpacking:BinPackingPP0", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

register(
    id="BinPacking-v3", entry_point="or_gym.envs.classic_or.binpacking:BinPackingPP1", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

register(
    id="BinPacking-v4", entry_point="or_gym.envs.classic_or.binpacking:BinPackingBW0", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

register(
    id="BinPacking-v5", entry_point="or_gym.envs.classic_or.binpacking:BinPackingBW1", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

# Newsvendor Envs
register(
    id="Newsvendor-v0", entry_point="or_gym.envs.classic_or.newsvendor:NewsvendorEnv", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

# Virtual Machine Packing Envs
register(id="VMPacking-v0", entry_point="or_gym.envs.classic_or.vmpacking:VMPackingEnv",additional_wrappers=(StepAPICompatibility.wrapper_spec(),))

register(
    id="VMPacking-v1", entry_point="or_gym.envs.classic_or.vmpacking:TempVMPackingEnv", additional_wrappers=(StepAPICompatibility.wrapper_spec(),)
)

# Vehicle Routing Envs
register(
    id="VehicleRouting-v0",
    entry_point="or_gym.envs.classic_or.vehicle_routing:VehicleRoutingEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

# TSP
register(id="TSP-v0", entry_point="or_gym.envs.classic_or.tsp:TSPEnv", additional_wrappers=(StepAPICompatibility.wrapper_spec(),))

register(id="TSP-v1", entry_point="or_gym.envs.classic_or.tsp:TSPDistCost",additional_wrappers=(StepAPICompatibility.wrapper_spec(),))

# Inventory Management Envs
register(
    id="InvManagement-v0",
    entry_point="or_gym.envs.supply_chain.inventory_management:InvManagementBacklogEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

register(
    id="InvManagement-v1",
    entry_point="or_gym.envs.supply_chain.inventory_management:InvManagementLostSalesEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

register(
    id="NetworkManagement-v0",
    entry_point="or_gym.envs.supply_chain.network_management:NetInvMgmtBacklogEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

register(
    id="NetworkManagement-v1",
    entry_point="or_gym.envs.supply_chain.network_management:NetInvMgmtLostSalesEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)

# Asset Allocation Envs
register(
    id="PortfolioOpt-v0",
    entry_point="or_gym.envs.finance.portfolio_opt:PortfolioOptEnv",
    additional_wrappers=(StepAPICompatibility.wrapper_spec(),),
)
