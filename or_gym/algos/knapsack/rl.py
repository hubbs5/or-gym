import numpy as np
import time
import gym
import or_gym

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune import grid_search

from or_gym.algos import rl_utils

def train_rl_knapsack(env_name, rl_config, max_episodes=1000):
    ray.init(ignore_reinit_error=True)
    # rl_utils.register_env(env_name) # Needed for Tune
    trainer = ppo.PPOTrainer(env=rl_utils.create_env(env_name), 
        config=rl_config)

    rewards = []
    eps, eps_total = [], []
    training = True
    batch = 0
    t_start = time.time()
    while training:
        t_batch = time.time()
        results = trainer.train()
        rewards.append(results['episode_reward_mean'])
        eps_total.append(results['episodes_total'])
        batch += 1
        t_end = time.time()
        if eps_total[-1] >= max_episodes:
            training = False
            break
        if batch % 10 == 0:
            t = t_end - t_batch
            t_tot = t_end - t_start
            print("\rEpisode: {}\tMean Rewards: {:.1f}\tEpisodes/sec: {:.2f}s\tTotal Time: {:.1f}s".format(
                eps_total[-1], rewards[-1], eps[-1]/t, t_tot), end="")
    ray.shutdown()
    print("Total Training Time: {:.1f}s\t".format(t_end - t_start))
    return trainer, np.array(rewards), np.array(eps_total)


# ModelCatalog.register_custom_model("my_model", rl_utils.CustomModel)




# x = tune.run(
#     "PPO",
#     stop={
#         "timesteps_total": 10000,
#     },
#     config={
#         "env": env_name,
#         "model": {
#             "custom_model": "my_model",
#         },
#         "env_config": {
#             "version": "Knapsack-v0"
#         },
#         "vf_share_layers": True,
#         "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
#         "num_workers": 1,  # parallelism
#     },
# )