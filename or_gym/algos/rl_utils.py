import ray
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
import gym
import or_gym
from datetime import datetime

# Get Ray to work with gym registry
def create_env(config, *args, **kwargs):
	if type(config) == dict:
		env_name = config['env']
	else:
		env_name = config
		print('Environment\t{}'.format(env_name))
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


def check_config(env_name, model_name=None, *args, **kwargs):
	if model_name is None:
		model_name = 'or_gym_tune'
	env = gym.make(env_name)
	try:
		vf_clip_param = env._max_rewards
	except AttributeError:
		vf_clip_param = 5000

	# TODO: Add grid search capabilities
	rl_config = {
		"env": env_name,
		"env_config": {
			# "version": env.spec.id.split('-')[-1]
			"reuse_actors":True
			},
		"vf_clip_param": vf_clip_param,
		"vf_share_layers": tune.grid_search([True, False]),
		"lr": tune.grid_search([1e-2, 1e-3, 1e-4, 1e-5]),
		"model": {
			"custom_model": model_name,
			"fcnet_activation": "elu",
			"fcnet_hiddens": [128, 128, 128]
			}
		}
	
	return rl_config

def register_env(env_name):
	env = create_env(env_name)
	tune.register_env(env_name, lambda env_name: env(env_name))

class FCModel(TFModelV2):

	def __init__(self, obs_space, action_space, num_outputs, model_config,
				 name):
		super(FCModel, self).__init__(
			obs_space, action_space, num_outputs,
			model_config, name)
		self.model = FullyConnectedNetwork(
			obs_space, action_space,
			num_outputs, model_config, name)
		self.register_variables(self.model.variables())

	def forward(self, input_dict, state, seq_lens):
		return self.model.forward(input_dict, state, seq_lens)

	def value_function(self):
		return self.model.value_function()

def tune_model(env_name, rl_config, model_name=None, algo='PPO'):
	if model_name is None:
		model_name = 'or_gym_tune'
	register_env(env_name)
	ray.init()
	ray.rllib.models.ModelCatalog.register_custom_model(model_name, FCModel)
	# Relevant docs: https://ray.readthedocs.io/en/latest/tune-package-ref.html
	results = tune.run(
		algo,
		stop={
			"timesteps_total": 1000000,
			"training_iteration": 10000 # Is this number of episodes?
		},
		config=rl_config
	)
	return results