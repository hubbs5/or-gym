import numpy as np
import gym
import or_gym
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork

import ray
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune import grid_search, register_env

from or_gym.envs.classic_or.knapsack import KnapsackEnv

tf = try_import_tf()

def create_env(config_env):
#     env = gym.make(config_env["version"])
    return KnapsackEnv()

class CustomModel(TFModelV2):
    """Example of a custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

register_env("Knapsack-v0", lambda config: create_env(config))
ray.init(ignore_reinit_error=True)
ModelCatalog.register_custom_model("my_model", CustomModel)
x = tune.run(
    "PPO",
    stop={
        "timesteps_total": 10000,
    },
    config={
        "env": "Knapsack-v0",  # or "corridor" if registered above
        "model": {
            "custom_model": "my_model",
        },
        "env_config": {
            "version": "Knapsack-v0"
#             "corridor_length": 5,
        },
        "vf_share_layers": True,
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 1,  # parallelism
    },
)