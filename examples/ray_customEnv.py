# -*- coding: utf-8 -*-
"""
Example for custom gym environment with ray Rllib 1.12.1.

Taken from https://docs.ray.io/en/latest/rllib/package_ref/env.html
"""

# __rllib-custom-gym-env-begin__
import gym

import ray
from ray.rllib.agents import ppo
from ray import tune


class SimpleCorridor(gym.Env):
    def __init__(self, config):
        self.end_pos = config["corridor_length"]
        self.cur_pos = 0
        self.action_space = gym.spaces.Discrete(2)  # right/left
        self.observation_space = gym.spaces.Discrete(self.end_pos)

    def reset(self):
        self.cur_pos = 0
        return self.cur_pos

    def step(self, action):
        if action == 0 and self.cur_pos > 0:  # move right (towards goal)
            self.cur_pos -= 1
        elif action == 1:  # move left (towards start)
            self.cur_pos += 1
        if self.cur_pos >= self.end_pos:
            return 0, 1.0, True, {}
        else:
            return self.cur_pos, -0.1, False, {}


ray.init()
config = {
    "env": SimpleCorridor,
    "env_config": {
        "corridor_length": 5,
    },
}

trainer = ppo.PPOTrainer(config=config)
for _ in range(3):
    print(trainer.train())
# __rllib-custom-gym-env-end__

# tune_config = {
#     'env': SimpleCorridor
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