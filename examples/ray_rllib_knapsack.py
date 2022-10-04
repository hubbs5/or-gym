# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.utils import try_import_tf
from gym import spaces
import or_gym.utils
from or_gym.envs.classic_or import KnapsackEnv
import numpy as np
 
tf_api, tf_original, tf_version = try_import_tf(error = True)


class KP0ActionMaskModel(TFModelV2):
     
    def __init__(self, obs_space, action_space, num_outputs,
        model_config, name, true_obs_shape=(11,),
        action_embed_size=5, *args, **kwargs):
        
        # true_obs_shape is going to match the size of the state. 
        # If we stick with our reduced KP, that will be a vector with 11 entries. 
        # The other value we need to provide is the action_embed_size, which is going to be the size of our action space (5)
         
        super(KP0ActionMaskModel, self).__init__(obs_space,
            action_space, num_outputs, model_config, name, 
            *args, **kwargs)
         
        self.action_embed_model = FullyConnectedNetwork(
            spaces.Box(0, 1, shape=true_obs_shape), 
                action_space, action_embed_size,
            model_config, name + "_action_embedding")
        self.register_variables(self.action_embed_model.variables())
 
    def forward(self, input_dict, state, seq_lens):
        
        # The actual masking takes place in the forward method where we unpack the mask, actions, and state from 
        # the observation dictionary provided by our environment. The state yields our action embeddings which gets 
        # combined with our mask to provide logits with the smallest value we can provide. 
        # This will get passed to a softmax output which will reduce the probability of selecting these actions to 0, 
        # effectively blocking the agent from ever taking these illegal actions.
        
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({
            "obs": input_dict["obs"]["state"]})
        # intent_vector = tf.expand_dims(action_embedding, 1)
        # action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)
        # inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        
        intent_vector = tf_api.expand_dims(action_embedding, 1)
        action_logits = tf_api.reduce_sum(avail_actions * intent_vector, axis=1)
        inf_mask = tf_api.maximum(tf_api.log(action_mask), tf_api.float32.min)
        
        return action_logits + inf_mask, state
 
    def value_function(self):
        return self.action_embed_model.value_function()


# Configuration for gym environment
# Not to be used for Online Knapsack
env_config = {'N': 5,
              'max_weight': 15,
              'item_weights': np.array([1, 12, 2, 1, 4]),
              'item_values': np.array([2, 4, 2, 1, 10]),
              'mask': True}

# Register the model for Rllib usage
ModelCatalog.register_custom_model('kp_mask', KP0ActionMaskModel)

# ATTENTION: Tune needs the base class, not an instance of the environment like we get from or_gym.make(env_name) to work with. So we need to pass this to register_env using a lambda function as shown below.
# tune.register_env('Knapsack-v0', lambda config: KnapsackEnv(env_config))


# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    # "env": 'Knapsack-v0',
    'env' : KnapsackEnv,
    # env config
    "env_config" : {
        'N': 5,
        'max_weight': 15,
        'item_weights': np.array([1, 12, 2, 1, 4]),
        'item_values': np.array([2, 4, 2, 1, 10]),
        'mask': True
        },
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 2,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf2",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "custom_model": "kp_mask", # Here we must use the custom model name taken in register process before
        "custom_model_config": {}
    },
    # Set up a separate evaluation worker set for the
    # `trainer.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}


# Headless start without ray.init()
trainer = PPOTrainer(config=config)

# # The real action masking logic: disable the agent to take action 0
# env = trainer.env_creator('Knapsack-v0')
# state = env.state
# state['action_mask'][0] = 0

# Train an agent for 1000 states and check if action 0 was not taken ever
# actions = np.array([trainer.compute_single_action(state) for i in range(10000)])
# print(any(actions==0))

# Use tune for hyperparameter tuning
# tune_config = {
#     'env': 'Knapsack-v0'
# }
# stop = {
#     'timesteps_total': 10000
# }
# results = tune.run(
#     'PPO', # Specify the algorithm to train
#     metric="score",
#     config=tune_config,
#     stop=stop
# ) 

# ray.shutdown()
