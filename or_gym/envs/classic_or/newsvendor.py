'''
Example taken from Balaji et al.
Paper: https://arxiv.org/abs/1911.10641
GitHub: https://github.com/awslabs/or-rl-benchmarks
'''
import gym
from gym import spaces
import itertools
import numpy as np
from collections.abc import Iterable
from or_gym.utils import assign_env_config

class NewsvendorEnv(gym.Env):
    '''
    Multi-Period Newsvendor with Lead Times

    The MPNV requires meeting stochastic demand by having sufficient
    inventory on hand to satisfy customers. The inventory orders are not
    instantaneous and have multi-period leadtimes. Additionally, there are
    costs associated with holding unsold inventory, however unsold inventory
    expires at the end of each period.

    Observation:
        Type: Box 
        State Vector: S = (p, c, h, k, mu, x_l, x_l-1)
        p = price
        c = cost
        h = holding cost
        k = lost sales penalty
        mu = mean of demand distribution
        x_l = order quantities in the queue

    Actions:
        Type: Box
        Amount of product to order.

    Reward:
        Sales minus discounted purchase price, minus holding costs for
        unsold product or penalties associated with insufficient inventory.

    Initial State:
        Parameters p, c, h, k, and mu, with no inventory in the pipeline.

    Episode Termination:
        Termination occurs after the maximum number of time steps is reached
        (40 by default).
    '''
    def __init__(self, *args, **kwargs):
        self.lead_time = 5
        self.max_inventory = 4000
        self.max_order_quantity = 2000
        self.step_limit = 40
        self.p_max = 100    # Max sale price
        self.h_max = 5      # Max holding cost
        self.k_max = 10     # Max lost sales penalty
        self.mu_max = 200   # Max mean of the demand distribution
        self.gamma = 1      # Discount factor
        assign_env_config(self, kwargs)

        self.obs_dim = self.lead_time + 5
        self.observation_space = spaces.Box(
            low=np.zeros(self.obs_dim),
            high=np.array(
                [self.p_max, self.p_max, self.h_max, self.k_max, self.mu_max] +
                [self.max_order_quantity] * self.lead_time),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.zeros(1), high=np.array([self.max_order_quantity]), 
            dtype=np.float32)

        self.reset()

    def _STEP(self, action):
        done = False
        order_qty = max(0, # Ensure order > 0
            min(action, self.max_inventory - self.state[5:].sum())) # Cap inventory
        demand = np.random.poisson(self.mu)
        inventory = self.state[5:]
        if self.lead_time == 0: # No lead time -> instant fulfillment
            inv_on_hand = order_qty
        else:
            inv_on_hand = inventory[0]
        sales = min(inv_on_hand, demand) * self.price
        excess_inventory = max(0, inv_on_hand - demand)
        short_inventory = max(0, demand - inv_on_hand)
        purchase_cost = excess_inventory * self.cost * order_qty * \
            self.gamma ** self.lead_time
        holding_cost = excess_inventory * self.h
        lost_sales_penalty = short_inventory * self.k
        reward = sales - purchase_cost - holding_cost - lost_sales_penalty

        # Update state, note inventory on hand expires at each time step
        new_inventory = np.zeros(self.lead_time)
        new_inventory[:-1] += inventory[1:]
        new_inventory[-1] += order_qty
        self.state = np.hstack([self.state[:5], new_inventory])

        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True
        if isinstance(reward, Iterable):
            # TODO: Sometimes reward is np.array with one entry
            reward = sum(reward)

        return self.state, reward, done, {}

    def _RESET(self):
        # Randomize costs
        self.price = max(1, np.random.rand() * self.p_max)
        self.cost = max(1, np.random.rand() * self.price)
        self.h = np.random.rand() * min(self.cost, self.h_max)
        self.k = np.random.rand() * self.k_max
        self.mu = np.random.rand() * self.mu_max
        self.state = np.zeros(self.obs_dim)
        self.state[:5] = np.array([self.price, self.cost, self.h,
            self.k, self.mu])

        self.step_count = 0

        return self.state

    def reset(self):
        return self._RESET()

    def step(self, action):
        return self._STEP(action)