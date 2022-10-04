# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 18:27:13 2022

@author: Philipp
"""

import gym
from ray.rllib.agents.ppo import PPOTrainer
# from or_gym.envs.supply_chain.inventory_management import InvManagementBacklogEnv

if __name__ == "__main__":
    # env = gym.make('InvManagement-v0')
    # print(env.reset())
    # print(env.step(1))
    trainer = PPOTrainer(
        config={
            "env": "CartPole-v0",
            # Parallelize environment rollouts.
            "num_workers": 3,
            "framework": "torch",
        }
    )
    