# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:07:26 2022

@author: Philipp
"""
import gym
from gym.spaces import Dict

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

from ray.rllib.models import ModelCatalog

from ray.rllib.agents.ppo import PPOTrainer, ppo
from ray.rllib.agents.dqn import DQNTrainer, dqn
from ray.rllib.agents import with_common_config

from ray import tune

import numpy as np

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class SimpleCorridor(gym.Env):
    """Corridor in which an agent must learn to move right to reach the exit.
    ---------------------
    | S | 1 | 2 | 3 | G |   S=start; G=goal; corridor_length=5
    ---------------------
    Possible actions to chose from are: 0=left; 1=right
    Observations are floats indicating the current field index, e.g. 0.0 for
    starting position, 1.0 for the field next to the starting position, etc..
    Rewards are -0.1 for all steps, except when reaching the goal (+1.0).
    """

    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # left and right
        self.observation_space = gym.spaces.Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

    def reset(self):
        """Resets the episode and returns the initial observation of the new one."""
        self.cur_pos = 0
        # Return initial observation.
        return [self.cur_pos]

    def step(self, action):
        """Takes a single step in the episode given `action`
        Returns:
            New observation, reward, done-flag, info-dict (empty).
        """
        # Walk left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Walk right.
        elif action == 1:
            self.cur_pos += 1
        # Set `done` flag when end of corridor (goal) reached.
        done = self.cur_pos >= self.end_pos
        # +1 when goal reached, otherwise -1.
        reward = 1.0 if done else -0.1
        return [self.cur_pos], reward, done, {}
    
class TFActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        # assert (
        #     isinstance(orig_space, Dict)
        #     and "action_mask" in orig_space.spaces
        #     and "observations" in orig_space.spaces
        # )

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


# class TorchActionMaskModel(TorchModelV2, nn.Module):
#     """PyTorch version of above ActionMaskingModel."""

#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         **kwargs,
#     ):
#         orig_space = getattr(obs_space, "original_space", obs_space)
#         assert (
#             isinstance(orig_space, Dict)
#             and "action_mask" in orig_space.spaces
#             and "observations" in orig_space.spaces
#         )

#         TorchModelV2.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name, **kwargs
#         )
#         nn.Module.__init__(self)

#         self.internal_model = TorchFC(
#             orig_space["observations"],
#             action_space,
#             num_outputs,
#             model_config,
#             name + "_internal",
#         )

#         # disable action masking --> will likely lead to invalid actions
#         self.no_masking = False
#         if "no_masking" in model_config["custom_model_config"]:
#             self.no_masking = model_config["custom_model_config"]["no_masking"]

#     def forward(self, input_dict, state, seq_lens):
#         # Extract the available actions tensor from the observation.
#         action_mask = input_dict["obs"]["action_mask"]

#         # Compute the unmasked logits.
#         logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

#         # If action masking is disabled, directly return unmasked logits
#         if self.no_masking:
#             return logits, state

#         # Convert action_mask into a [0.0 || -inf]-type mask.
#         inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
#         masked_logits = logits + inf_mask

#         # Return masked logits.
#         return masked_logits, state

#     def value_function(self):
#         return self.internal_model.value_function()
    
    
# main program
ModelCatalog.register_custom_model('tf_action_mask', TFActionMaskModel)

# with_common_config starts with the default config and then adds the parameter-given dict
# trainer_config  = with_common_config({
#     "env" : SimpleCorridor,
#     "env_config" : {
#         "corridor_length" : 5
#         }
# })

config  = {
    "env" : SimpleCorridor,
    "env_config" : {
        "corridor_length" : 5
        }
}

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update(config)
trainer = PPOTrainer(config=config)

# dqn_config = dqn.DEFAULT_CONFIG.copy()
# dqn_config.update(config)
# trainer = DQNTrainer(config=config)

for _ in range(5) :
    trainer.train()
print("Training finished")
    
tune_config = {
    'env': SimpleCorridor,
    "env_config" : {
        "corridor_length" : 5
        },
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf2",
    "evaluation_num_workers": 1,
    # Model selection
    "model": {
        "custom_model": "tf_action_mask", # Here we must use the custom model name taken in register process before
        # "custom_model_config": {
        #     "no_masking" : False
        #     }
    },
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
    'log_level': 'WARN',
}
stop = {
    'timesteps_total': 10000
}
results = tune.run(
    # 'PPO', 
    'DQN',
    metric="score",
    config=tune_config,
    stop=stop
)  