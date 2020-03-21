# Get Ray to work with gym registry
def create_env(env_name):
    if env_name == 'Knapsack-v0':
        from or_gym.envs.classic_or.knapsack import KnapsackEnv as env
    elif env_name == 'Knapsack-v1':
        from or_gym.envs.classic_or.knapsack import BoundedKnapsackEnv as env
    elif env_name == 'Knapsack-v2':
        from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv as env
    elif env_name == 'BinPacking-v0':
    	from or_gym.envs.classic_or.binpacking import BinPackingEnv as env
    elif env_name == 'BinPacking-v1':
    	raise NotImplementedError('{} not yet implemented.'.format(env_name))
    	from or_gym.envs.classic_or.binpacking import BinPackingEnv as env
    elif env_name == 'BinPacking-v2':
    	raise NotImplementedError('{} not yet implemented.'.format(env_name))
    	from or_gym.envs.classic_or.binpacking import BinPackingEnv as env
    elif env_name == 'VMPackgin-v0':
    	from or_gym.envs.classic_or.vm_packing import VMPackingEnv as env
    elif env_name == 'VMPacking-v1':
    	from or_gym.envs.classic_or.vm_packing import TempVMPackingEnv as env
    elif env_name == 'PortfolioOpt-v0':
    	from or_gym.envs.classic_or.portfolio_opt import PortfolioOptEnv as env
    elif env_name == 'TSP-v0':
    	raise NotImplementedError('{} not yet implemented.'.format(env_name))
    	from or_gym.envs.classic_or.tsp import TSPEnv as env
    elif env_name == 'VRP-v0':
    	raise NotImplementedError('{} not yet implemented.'.format(env_name))
    elif env_name == 'NewsVendor-v0':
    	raise NotImplementedError('{} not yet implemented.'.format(env_name))
    else:
        raise NotImplementedError('Environment {} not recognized.'.format(env_name))
    return env