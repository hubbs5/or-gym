import numpy as np

def assign_env_config(self, kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)
    if hasattr(self, 'env_config'):
        for key, value in self.env_config.items():
            # Check types based on default settings
            if hasattr(self, key):
                if type(getattr(self,key)) == np.ndarray:
                    setattr(self, key, value)
                else:
                    setattr(self, key,
                        type(getattr(self, key))(value))
            else:
                raise AttributeError(f"{self} has no attribute, {key}")

# Get Ray to work with gym registry
def create_env(config, *args, **kwargs):
  if type(config) == dict:
    env_name = config['env']
  else:
    env_name = config
  
  if env_name.lower() == 'knapsack-v0':
    from or_gym.envs.classic_or.knapsack import KnapsackEnv as env
  elif env_name.lower() == 'knapsack-v1':
    from or_gym.envs.classic_or.knapsack import BinaryKnapsackEnv as env
  elif env_name.lower() == 'knapsack-v2':
    from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv as env
  elif env_name.lower() == 'knapsack-v3':
    from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv as env
  elif env_name.lower() == 'binpacking-v0':
    from or_gym.envs.classic_or.binpacking import BinPackingEnv as env
  elif env_name.lower() == 'binpacking-v1':
    from or_gym.envs.classic_or.binpacking import BinPackingLW1 as env
  elif env_name.lower() == 'binpacking-v2':
    from or_gym.envs.classic_or.binpacking import BinPackingPP0 as env
  elif env_name.lower() == 'binpacking-v3':
    from or_gym.envs.classic_or.binpacking import BinPackingPP1 as env
  elif env_name.lower() == 'binpacking-v4':
    from or_gym.envs.classic_or.binpacking import BinPackingBW0 as env
  elif env_name.lower() == 'binpacking-v5':
    from or_gym.envs.classic_or.binpacking import BinPackingBW1 as env
  elif env_name.lower() == 'vmpacking-v0':
    from or_gym.envs.classic_or.vmpacking import VMPackingEnv as env
  elif env_name.lower() == 'vmpacking-v1':
    from or_gym.envs.classic_or.vmpacking import TempVMPackingEnv as env
  elif env_name.lower() == 'portfolioopt-v0':
    from or_gym.envs.finance.portfolio_opt import PortfolioOptEnv as env
  elif env_name.lower() == 'tsp-v0':
    from or_gym.envs.classic_or.tsp import TSPEnv as env
  elif env_name.lower() == 'tsp-v1':
    from or_gym.envs.classic_or.tsp import TSPDistCost as env
  elif env_name.lower() == 'vehiclerouting-v0':
    from or_gym.envs.classic_or.vehicle_routing import VehicleRoutingEnv as env
  elif env_name.lower() == 'newsvendor-v0':
    from or_gym.envs.classic_or.newsvendor import NewsvendorEnv as env
  elif env_name.lower() == 'invmanagement-v0':
    from or_gym.envs.supply_chain.inventory_management import InvManagementBacklogEnv as env
  elif env_name.lower() == 'invmanagement-v1':
    from or_gym.envs.supply_chain.inventory_management import InvManagementLostSalesEnv as env
  elif env_name.lower() == 'networkmanagement-v0':
    from or_gym.envs.supply_chain.network_management import NetInvMgmtBacklogEnv as env
  elif env_name.lower() == 'networkmanagement-v1':
    from or_gym.envs.supply_chain.network_management import NetInvMgmtLostSalesEnv as env
  else:
    raise NotImplementedError('Environment {} not recognized.'.format(env_name))
  return env
