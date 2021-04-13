import numpy as np

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        # print(self.env_config)
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                setattr(self, key, value)
                
# Get Ray to work with gym registry
def create_env(config, *args, **kwargs):
	if type(config) == dict:
		env_name = config['env']
	else:
		env_name = config
	if env_name == 'Knapsack-v0':
		from or_gym.envs.classic_or.knapsack import KnapsackEnv as env
	elif env_name == 'Knapsack-v1':
		from or_gym.envs.classic_or.knapsack import BinaryKnapsackEnv as env
	elif env_name == 'Knapsack-v2':
		from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv as env
	elif env_name == 'Knapsack-v3':
		from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv as env
	elif env_name == 'BinPacking-v0':
		from or_gym.envs.classic_or.binpacking import BinPackingEnv as env
	elif env_name == 'BinPacking-v1':
		from or_gym.envs.classic_or.binpacking import BinPackingLW1 as env
	elif env_name == 'BinPacking-v2':
		from or_gym.envs.classic_or.binpacking import BinPackingPP0 as env
	elif env_name == 'BinPacking-v3':
		from or_gym.envs.classic_or.binpacking import BinPackingPP1 as env
	elif env_name == 'BinPacking-v4':
		from or_gym.envs.classic_or.binpacking import BinPackingBW0 as env
	elif env_name == 'BinPacking-v5':
		from or_gym.envs.classic_or.binpacking import BinPackingBW1 as env
	elif env_name == 'VMPacking-v0':
		from or_gym.envs.classic_or.vmpacking import VMPackingEnv as env
	elif env_name == 'VMPacking-v1':
		from or_gym.envs.classic_or.vmpacking import TempVMPackingEnv as env
	elif env_name == 'PortfolioOpt-v0':
		from or_gym.envs.finance.portfolio_opt import PortfolioOptEnv as env
	elif env_name == 'TSP-v0':
		raise NotImplementedError('{} not yet implemented.'.format(env_name))
	elif env_name == 'VehicleRouting-v0':
		from or_gym.envs.classic_or.vehicle_routing import VehicleRoutingEnv as env
	elif env_name == 'VehicleRouting-v1':
		from or_gym.envs.classic_or.vehicle_routing import VehicleRoutingEnv as env
	elif env_name == 'NewsVendor-v0':
		from or_gym.envs.classic_or.newsvendor import NewsvendorEnv as env
	elif env_name == 'InvManagement-v0':
		from or_gym.envs.supply_chain.inventory_management import InvManagementBacklogEnv as env
	elif env_name == 'InvManagement-v1':
		from or_gym.envs.supply_chain.inventory_management import InvManagementLostSalesEnv as env
	elif env_name == 'InvManagement-v2':
		from or_gym.envs.supply_chain.network_management import NetInvMgmtBacklogEnv as env
	elif env_name == 'InvManagement-v3':
		from or_gym.envs.supply_chain.network_management import NetInvMgmtLostSalesEnv as env
	else:
		raise NotImplementedError('Environment {} not recognized.'.format(env_name))
	return env
