import numpy as np
import gym
from gym import spaces
from abc import ABC, abstractmethod
from or_gym import utils

class BaseSchedEnv(gym.Env, ABC):
    '''
    Base scheduling environment. Additional models can be built on 
    top of this class.

    Parameters:
        time_limit: int, number of hours in the episode.
        n_stages: int, number of processing stages.
        n_fin_products: int, number of finished products that can be sold.
        n_int_products: int, number of intermediate products that are 
            converted into finished products.
        product_ids: array, identifier for each product.
        _run_rate: int, placeholder value for metric tons produced per hour.
        run_rates: dict, maps products to run rate to allow for variation. 
            Uses _run_rate by default.
        _product_value: int, default value of each metric ton of product.
        product_values: dict, maps products to their values. Uses 
            _product_value by default.
        ship_by_time: int, time to ship orders each day. If orders are due on
            a given day, but material isn't available to ship them, they will
            be marked late and penalties may be applied.
        init_inventory: array, initial inventory to begin each episode.
        _cure_time: int, hours a product must sit (cool, degas, etc.) before
            it can be shipped or processed by the next stage.
        cure_times: dict, maps products to cure times. Uses _cure_time value
            for all products by default.
        _holding_cost: int, unit cost for holding inventory.
        holding_costs: dict, maps products to specific holding costs. Uses
            _holding_cost by default.
        _converstion_rate: float, used to convert input products to output
            quantities.
        conversion_rates: dict, converts input products to output quantities.
            Uses _conversion_rate by default.
    '''
    def __init__(self, *args, **kwargs):
        self.time_limit = 365 * 24 # Hours
        self.n_fin_products = 10
        self.n_int_products = 0
        self.product_ids = np.arange(self.n_fin_products + self.n_int_products)
        self.init_inventory = np.zeros(self.n_fin_products + self.n_int_products)
        self.n_stages = 1
        self._run_rate = 10 # MT/hour
        self._product_value = 10 # $/MT
        self._min_production_qty = 100 # MT
        self.ship_by_time = 17 # Orders ship by 5 pm each day
        self._cure_time = 24 # Hours
        self._holding_cost = 1 # $/MT
        self._conversion_rate = 1
        
        self.run_rates = {i: self._run_rate 
            for i in self.product_ids}
        self.product_values = {i: self._product_value
            for i in self.product_ids}
        self.min_product_qtys = {i: self._min_production_qty
            for i in self.product_ids}
        self.cure_times = {i: self._cure_time 
            for i in self.product_ids}
        self.holding_costs = {i: self._holding_cost
            for i in self.product_ids}
        self.conversion_rates = {i: self._conversion_rate
            for i in self.product_ids}

        self._check_unique_prods()

    def _check_unique_prods(self):
        # Product IDs must be unique.
        _, count = np.unique(self.product_ids, return_counts=True)
        assert count.max() == 1, "Non-unique products found: {}".format(
            self.product_ids[np.where(count>1)])

    def _STEP(self, action):
        pass

    def _RESET(self):
        pass

    def calculate_reward(self):
        pass

    def get_demand(self):
        pass

    def get_state(self):
        pass

    @abstractmethod
    def step(self, action):
        raise NotImplementedError("step() method not implemented.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset() method not implemented.")

class SingleStageSchedEnv(BaseSchedEnv):

    def __init__(self, *args, **kwargs):
        super().__init__()
        utils.assign_env_config(self, kwargs)

        self.reset()

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()