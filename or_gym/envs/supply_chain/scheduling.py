import numpy as np
import gym
from gym import spaces
from abc import ABC, abstractmethod
from collections import namedtuple, deque
from collections.abc import Iterable
from operator import attrgetter
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
        _run_rate: int, placeholder value for units produced per hour.
        run_rates: dict, maps products to run rate to allow for variation. 
            Uses _run_rate by default.
        _product_value: int, default value of each units of product.
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
        order_qty: int, fixed quantity for orders.
    '''
    def __init__(self, *args, **kwargs):
        self.simulation_days = 365
        self.time_limit = self.simulation_days * 24 # Hours
        self.n_fin_products = 10
        self.n_int_products = 0
        self.product_ids = np.arange(self.n_fin_products + self.n_int_products)
        self.init_inventory = np.zeros(self.n_fin_products + self.n_int_products)
        self.n_stages = 1
        self._run_rate = 10 # Units/hour
        self._product_value = 10 # $/Unit
        self._min_production_qty = 100 # Units
        self.ship_by_time = 24 # Orders ship by midnight each day
        self._cure_time = 24 # Hours
        self._holding_cost = 1 # $/Units
        self._conversion_rate = 1
        self.avg_lead_time = 7 # Days
        
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

        self.order_book_cols = ['Num', 'ProductID', 'CreateDate', 'DueDate', 
            'Value', 'Shipped', 'ShipDate', 'OnTime']
        self.ob_col_idx = {j: i for i, j in enumerate(self.order_book_cols)}

        self.sched_cols = ['Num', 'ProductID', 'Stage', 'Line', 
            'StartTime', 'EndTime', 'Quantity', 'Completed']
        self.sched_col_idx = {j: i for i, j in enumerate(self.sched_cols)}

        self._check_unique_prods()
        self._init_demand = False
        self._initialize_demand_model()

    def _check_unique_prods(self):
        # Product IDs must be unique.
        _, count = np.unique(self.product_ids, return_counts=True)
        assert count.max() == 1, "Non-unique products found: {}".format(
            self.product_ids[np.where(count>1)])

    def _STEP(self, action):
        pass

    def _RESET(self):
        self.production_deque = deque()
        pass

    def calculate_reward(self):
        pass

    def _initialize_demand_model(self):
        # Only initialize at beginning of model
        if self._init_demand is False:
            return None
        self.mean_total_demand = np.mean([i
            for i in self.run_rates.values()]) * self.time_limit
        
        self.product_demand = self._get_product_demand()
        self._seasonal_offset = np.random.rand() * 2 * np.pi
        _sin = np.sin(np.linspace(0, 2*np.pi, self.simulation_days)
            + self._seasonal_offset) + 1
        self._p_seas = _sin / _sin.sum()
        self._init_demand = True

    def _get_product_demand(self):
        # Randomly provides a percentage of total demand to each finished
        # product.
        s = np.random.normal(size=self.n_fin_products)
        shares = utils.softmax(s)
        return np.round(shares * self.mean_total_demand, 0).astype(int)

    def _run_demand_model(self):
        # Base model calculates mean run rate for the products, multiplies
        # this by the number of hours in the simulation, and uses this as
        # the mean, total demand for the each episode. A fraction of the
        # total demand is then split among the various final products. Time
        # series are made by sampling orders from a normalized sine wave to
        # simulate seasonality. Random values (e.g. demand shares) are
        # fixed whenever or_gym.make() is called, and are preserved during each
        # call to reset().
        # Returns order_book object containing demand data for the episode.
        self._initialize_demand_model()
        order_book = np.zeros((self.product_demand.sum(), 
            len(self.order_book_cols)))
        order_book[:, self.ob_col_idx['Num']] = np.arange(
            self.product_demand.sum())
        prods = np.hstack([np.repeat(i, j)
            for i, j in zip(self.product_ids, self.product_demand)])
        order_book[:, self.ob_col_idx['ProductID']] = prods
        due_dates = np.hstack([np.choice(np.arange(0, self.simulation_days),
            p=self._p_seas, size=i)
            for i in self.product_demand])
        order_book[:, self.ob_col_idx['DueDate']] = due_dates
        order_book[:, self.ob_col_idx['CreateDate']] = due_dates - \
            np.random.poisson(lam=self.avg_lead_time, 
                size=self.product_demand.sum())

        return order_book

    def get_demand(self):
        self.order_book = self._run_demand_model()

    def get_state(self):
        pass

    @abstractmethod
    def step(self, action):
        raise NotImplementedError("step() method not implemented.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset() method not implemented.")

class SingleStageSchedEnv(BaseSchedEnv):
    '''
    This is the simplest scheduling environment where the agent needs to 
    manage a single production line with fixed production sizes and rates 
    to meet demand.

    The only action available to the agent is what product to produce next.
    This selection is appended to the end of the schedule and production will
    begin on this next product when it is scheduled.

    Actions:
        Type: Discrete
        0: Produce product 0
        1: Produce product 1
        2: ... 

    Observations:
        Type: Dictionary
        production_state:
            Type: Box
            0: Current time
            1: Schedule time
            2: Inventory of product 0
            3: Inventory of product 1
            4: ... 
        demand_state:
            Type: Box
        forecast_state:
            Type: Box
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        utils.assign_env_config(self, kwargs)

        self.observation_space = None
        self.action_space = None

        self.reset()

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()