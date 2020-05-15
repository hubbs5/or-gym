import gym
from gym import spaces
from or_gym.utils.env_config import *
import numpy as np

class BaseSchedEnv(gym.Env):
    
    def __init__(self, *args, **kwargs):
        
        self.num_products = 3
        self.step_limit = 10
        self.transition_prob = [0.5, 0.5]
        self.sales_price = np.ones(self.num_products)
        self.holding_costs = np.ones(self.num_products) * 2
        self.production_rate = np.ones(self.num_products) * 10
        self.max_inventory = 100
        self.action_setting = 'discrete'
        assign_env_config(self, kwargs)
        # self.transition_matrix = self.generate_transition_matrix()
        self.transition_matrix = np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]])
        self.obs_dim = self.num_products * 2 + 3
        # Reference to see which columns in state corresponds to different inputs
        self.state_indices = {
            'time': 0,
            'prev_actions': [1,2],
            'inventory': 
                {'product_' + str(j): i 
                 for i, j in enumerate(np.arange(self.num_products), 3)},
            'demand': 
                {'product_' + str(j): i 
                 for i, j in enumerate(np.arange(self.num_products), 
                    self.num_products + 3)}
        }
        
        self.reset()

    def _split_action(self, action):
        if self.action_setting == 'continuous':
            # If multiple actions are tied for max value, randomly break tie
            rate = np.max(action)
            if len(np.where(action==rate)[0]) > 1:
                product = np.random.choice(np.where(action==rate)[0])
            else:
                product = np.argmax(action)
        else:
            product = int(action)
            rate = 1 #self.production_rate[product]
        return product, rate
        
    def step(self, action):
        product, rate = self._split_action(action)
        done = False
        t = self.current_step
        self.inventory[self.last_action[0]] += self.planned_production
        # To be carried forward to next step
        valid_action = self.transition_matrix[self.last_action[0], product]
        self.planned_production = valid_action * rate * self.production_rate[product]
        # Large negative reward for transitions with 0
        reward = -100 if valid_action == 0 else 0
        
        # Sell products
        sales_qty = [self.demand[t, i] if j else self.inventory[i] 
            for i, j in enumerate(self.inventory>=self.demand[t])]
        self.inventory = np.where(self.inventory - self.demand[t]>=0, 
            self.inventory-self.demand[t], 0)
        self.inventory = np.where(self.inventory>self.max_inventory, 
            self.max_inventory, self.inventory)
        self.last_action = [product, rate]
        reward += np.dot(sales_qty, self.sales_price) - \
            np.dot(self.inventory, self.holding_costs)
        
        self.current_step += 1
        if self.current_step >= self.step_limit:
            done = True
        self.state = self._update_state()
        
        return self.state, reward, done, {}
        
    def reset(self):
        self.demand = self.generate_demand()
        self.inventory = np.zeros(self.num_products)
        # Randomly initialize schedule
        prev_prod = np.random.choice(np.arange(self.num_products))
        if self.action_setting == 'continuous':
            rate = np.random.uniform(low=0, high=self.production_rate[prev_prod])
        else:
            rate = self.production_rate[prev_prod]
        self.planned_production = rate
        self.last_action = [prev_prod, rate]
        self.current_step = 0
                
        return self._update_state()
    
    def generate_demand(self):
        demand = np.zeros((self.step_limit+1, self.num_products))
        for i in range(self.step_limit):
            for j in range(self.num_products):
                if i % self.num_products == j:
                    demand[i, j] += 5
        return demand
    
    def sample_action(self):
        return self.action_space.sample()

    def generate_transition_matrix(self):
        tm = np.random.choice([0, 1], p=self.transition_prob, 
            size=(self.num_products, self.num_products))
        tm = tm + tm.T + np.identity(self.num_products)
        tm = np.where(tm>1, 1, tm)
        # Ensure at least one viable transition for each product
        idx = np.where(tm.sum(axis=1) < 2)[0]
        while len(idx) > 0:
            tm[idx] += np.random.choice([0, 1], 
                p=self.transition_prob, size=self.num_products)
            tm = np.where(tm>1, 1, tm)
            idx = np.where(tm.sum(axis=1) < 2)[0]
        return tm

class DiscreteSchedEnv(BaseSchedEnv):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_setting = 'discrete'
        self.action_space = spaces.Discrete(self.num_products, dtype=np.int16)
        self.observation_space = spaces.Box(0, 
            self.max_inventory, shape=(self.obs_dim,), dtype=np.int16)

        self.reset()

    def _update_state(self):
        t = self.current_step
        self.state = np.hstack([t, self.last_action, self.inventory, 
            self.demand[t]])
            
        return self.state

class MaskedDiscreteSchedEnv(BaseSchedEnv):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_setting = 'discrete'
        self.action_space = spaces.Discrete(self.num_products, dtype=np.int16)
        self.observation_space = spaces.Dict({
            'action_mask': spaces.Box(0, 1, shape=(self.num_products,), 
                dtype=np.int16),
            'avail_actions': spaces.Box(0, 1, shape=(self.num_products,), 
                dtype=np.int16),
            'state': spaces.Box(0, 
                self.max_inventory, shape=(self.obs_dim,),
                dtype=np.float32)
        })

        self.reset()

    def _update_state(self):
        t = self.current_step
        state = np.hstack([t, self.last_action, self.inventory, 
            self.demand[t]])
        idx = self.last_action[0]
        action_mask = self.transition_matrix[idx]
        self.state = {'action_mask': action_mask,
                      'avail_actions': np.ones(self.num_products),
                      'state': state}
            
        return self.state

class ContSchedEnv(BaseSchedEnv):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_setting = 'continuous'
        self.action_space = spaces.Box(
            low=np.zeros(self.num_products),
            high=np.ones(self.num_products), 
            dtype=np.float32)
        self.observation_space = spaces.Box(0, self.max_inventory, 
            shape=(self.obs_dim,), dtype=np.float32)

        self.reset()            

    def _update_state(self):
        t = self.current_step
        self.state = np.hstack([t, self.last_action, 
            self.inventory, self.demand[t]])            
        return self.state

class MaskedContSchedEnv(BaseSchedEnv):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.action_setting = 'continuous'
        self.action_space = spaces.Box(
            low=np.zeros(self.num_products), 
            high=np.ones(self.num_products), 
            dtype=np.float32)
        self.observation_space = spaces.Dict({
            'action_mask': spaces.Box(0, 1, shape=(self.num_products,), 
                dtype=np.int16),
            'avail_actions': spaces.Box(0, 1, shape=(self.num_products,),
                dtype=np.int16),
            'state': spaces.Box(0, self.max_inventory, shape=(self.obs_dim,),
                dtype=np.float32)
        })

        self.reset()

    def _update_state(self):
        t = self.current_step
        state = np.hstack([t, self.last_action, 
            self.inventory, self.demand[t]])
        idx = self.last_action[0]
        action_mask = self.transition_matrix[idx]
        self.state = {'action_mask': action_mask,
                      'avail_actions': np.ones(self.num_products),
                      'state': state}
            
        return self.state