import ray
from ray import tune
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from gym import spaces
import or_gym
from datetime import datetime
from copy import deepcopy


tf = try_import_tf()

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
	else:
		raise NotImplementedError('Environment {} not recognized.'.format(env_name))
	return env

def set_config(default_config, config_dict=None):
    config = deepcopy(default_config)
    if type(config_dict) == dict:
        for k in config.keys():
            if k in config_dict.keys():
                if type(config[k]) == dict:
                    for m in config[k].keys():
                        if m in config_dict.keys():
                            config[k][m] = config_dict[m]
                else:
                    config[k] = config_dict[k]
            else:
                continue
                
    return config

def check_config(env_name, model_name=None, *args, **kwargs):
	if model_name is None:
		model_name = 'or_gym_tune'
	env = or_gym.make(env_name)
	try:
		vf_clip_param = env._max_rewards
	except AttributeError:
		vf_clip_param = 10

	# TODO: Add grid search capabilities
	rl_config = {
		"env": env_name,
		"num_workers": 2,
		"env_config": {
			'mask': True
			},
		# "lr": 1e-5,
		# "entropy_coeff": 1e-4,
		"vf_clip_param": vf_clip_param,
		"lr": tune.grid_search([1e-4, 1e-5]), #1e-6, 1e-7]),
		"entropy_coeff": tune.grid_search([1e-2, 1e-3]), #, 1e-4]),
		# "critic_lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
		# "actor_lr": tune.grid_search([1e-3, 1e-4, 1e-5]),
		# "lambda": tune.grid_search([0.95, 0.9]),
		"kl_target": tune.grid_search([0.01, 0.03]),
		# "sgd_minibatch_size": tune.grid_search([128, 512, 1024]),
		# "train_batch_size": tune.grid_search([])
		"model": {
			"vf_share_layers": False,
			# "custom_model": model_name,
			"fcnet_activation": "elu",
			"fcnet_hiddens": [128, 128, 128]
			}
	}
	
	return rl_config

def register_env(env_name):
	env = create_env(env_name)
	tune.register_env(env_name, lambda env_name: env(env_name))

class FCModel(TFModelV2):
	'''Fully Connected Model'''
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

class KP0ActionMaskModel(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, true_obs_shape=(401,), action_embed_size=200,
        *args, **kwargs):
        super(KP0ActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, *args, **kwargs)
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
        
    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]
        })
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state
    
    def value_function(self):
        return self.action_embed_model.value_function()


class KP1ActionMaskModel(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, true_obs_shape=(3, 201), action_embed_size=200,
        *args, **kwargs):
        super(KP1ActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, *args, **kwargs)
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
        
    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]
        })
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state
    
    def value_function(self):
        return self.action_embed_model.value_function()

class VMActionMaskModel(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, true_obs_shape=(51, 3), action_embed_size=50,
        *args, **kwargs):
        super(VMActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, *args, **kwargs)
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
        
    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]
        })
        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state
    
    def value_function(self):
        return self.action_embed_model.value_function()

def tune_model(env_name, rl_config, model_name=None, algo='PPO'):
	if model_name is None:
		model_name = 'or_gym_tune'
	register_env(env_name)
	ray.init()
	if "VMPacking" in rl_config["env"]:
		ray.rllib.models.ModelCatalog.register_custom_model(model_name, VMActionMaskModel)
	elif "Knapsack-v0" in rl_config["env"]:
		ray.rllib.models.ModelCatalog.register_custom_model(model_name, KP0ActionMaskModel)
	elif "Knapsack-v1" in rl_config["env"] or "Knapsack-v2" in rl_config["env"]:
		ray.rllib.models.ModelCatalog.register_custom_model(model_name, KP1ActionMaskModel)
	# else:
		# ray.rllib.models.ModelCatalog.register_custom_model(model_name, FCModel)
	# Relevant docs: https://ray.readthedocs.io/en/latest/tune-package-ref.html
	results = tune.run(
		algo,
		checkpoint_freq=100,
		checkpoint_at_end=True,
		queue_trials=True,
		stop={
			"training_iteration": 500,
			"episode_reward_mean": 2696
		},
		config=rl_config,
		reuse_actors=True
	)
	return results