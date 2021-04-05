import numpy as np
import gym
from gym import spaces
from abc import ABC, abstractmethod, ABCMeta
from collections import namedtuple, deque, OrderedDict
from collections.abc import Iterable
from operator import attrgetter
from or_gym import utils

class BaseSchedEnv(gym.Env):
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
        min_schedule_length: int, minimum time the schedule needs to be
            maintained in hours.
        late_penalty: float, fraction of value that an order declines each
            day it is late. This penalty is assessed every day, so can exceed
            the value of the order.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, *args, **kwargs):
        self.simulation_days = 365
        self.time_limit = self.simulation_days * 24 # Hours
        self.n_fin_products = 10 # Finished products
        self.n_int_products = 0  # Intermediate products
        self.tot_products = self.n_fin_products + self.n_int_products
        self.product_ids = np.arange(self.tot_products)
        self.init_inventory = np.ones(self.tot_products) * 100
        self.n_stages = 1
        self._run_rate = 10 # Units/hour
        self._product_value = 10 # $/Unit
        self._min_production_qty = 100 # Units
        self.ship_by_time = 24 # Orders ship by midnight each day
        self._cure_time = 24 # Hours
        self._holding_cost = 1 # $/Units
        self._conversion_rate = 1 # Applicable for multi-stage models
        self.avg_lead_time = 7 # Days
        self.min_schedule_length = 7 * 24 # Hours
        self.late_penalty = 0.1
        self._transition_matrix = np.random.choice(np.arange(13), 
            size=(self.tot_products, self.tot_products))
        np.fill_diagonal(self._transition_matrix, 0)
        
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
        self.prod_inv_idx = {j: i 
            for i, j in enumerate(self.product_ids)}
        self.transition_dict = {(j, l): self._transition_matrix[i, k] 
            for k, l in enumerate(self.product_ids) 
            for i, j in enumerate(self.product_ids)}

        self.order_book_cols = ['Number', 'ProductID', 'CreateTime', 'DueTime', 
            'Quantity', 'Value', 'Shipped', 'ShipTime', 'OnTime']
        self.ob_col_idx = {j: i for i, j in enumerate(self.order_book_cols)}

        self.sched_cols = ['Number', 'ProductID', 'Stage', 'Line', 
            'ProdStartTime', 'ProdEndTime', 'ProdReleaseTime', 'Quantity', 
            'OffGrade', 'Completed']
        self.sched_col_idx = {j: i for i, j in enumerate(self.sched_cols)}

        self._check_unique_prods()
        self._init_demand = False
        self._initialize_demand_model()
        self._vectorize_mappings()
        self._build_hc_array()
        self._stage_line_dict = self._init_stage_line_dict()

    def _check_unique_prods(self):
        # Product IDs must be unique.
        _, count = np.unique(self.product_ids, return_counts=True)
        assert count.max() == 1, "Non-unique products found: {}".format(
            self.product_ids[np.where(count>1)])

    def _build_hc_array(self):
        self._holding_cost_array = np.array([v for v in self.holding_costs.values()])

    def _STEP(self, action):
        if not isinstance(action, Iterable):
            action = np.array([action])

        done = False
        reward = 0
        self.append_schedules(action)
        # Get next prod end time and the latest prod start time from scheds
        self._next_end_time = self._get_next_end_time()
        self._latest_start_time = self._get_latest_start_time()
        
        # While the schedule extends beyond the minimum  schedule length,
        # the environment pushes new product through the reactors, updates
        # inventory, and ships orders.
        while self._next_end_time > self.env_time + self.min_schedule_length:
            if len(self.production_deque) < self._action_inputs:
                self.production_deque = self._get_next_production_starts()
            
            # Begin production
            try:
                while self.production_deque[0].ProdStartTime == self.env_time:
                    self._start_production(self.production_deque[0])
                    self.production_deque.popleft()
            except IndexError:
                pass
            self.production_release_deque = deque(
                sorted(self.production_release_deque, 
                key=attrgetter('ProdReleaseTime')))
            
            # Release products to inventory
            try:
                while self.production_release_deque[0].ProdReleaseTime == self.env_time:
                    self._book_inventory(self.production_release_deque[0])
                    self.production_release_deque.popleft()
            except IndexError:
                pass
            
            # Ship orders to fulfill demand
            if self.env_time % self.ship_by_time == 0:
                self.ship_orders()
            if self.run_maintentance_model():
                break

            # Get rewards every 24-hours
            if self.env_time % 24 == 0:
                self._holding_costs = self._calc_holding_cost()
                reward += self._calc_reward()

            self.env_time += 1

            if self.env_time >= self.time_limit:
                done = True
                break
        
        self.state = self.get_state()
        info = {}
        return self.state, reward, done, info

    def _get_latest_start_time(self):
        # Returns the latest start times from all schedules unless schedules
        # are None, then returns current time.
        return np.array([v1[-1, self.sched_col_idx['ProdStartTime']]
            if v1 is not None else self.env_time
            for k, v in self.schedules.items()
            for k1, v1 in v.items()]).min()

    def _get_next_end_time(self):
        # Returns earliest production end time across schedules
        return np.array([v1[-1, self.sched_col_idx['ProdEndTime']]
            if v1 is not None else self.env_time
            for k, v in self.schedules.items()
            for k1, v1 in v.items()])

    def _start_production(self, prod_tuple):
        '''
        Begins production based on the schedule. 
        TODO: Add raw material checks for multi-stage models.
        '''
        self.current_production[prod_tuple.Stage][prod_tuple.Line] = prod_tuple
        self.production_release_deque.append(prod_tuple)

    def ship_orders(self):
        shipped_orders = None
        late_orders = None
        # Orders to Ship
        ots = self.order_book[np.logical_and(
            self.order_book[:,self.ob_col_idx['Shipped']]==0,
            self.order_book[:,self.ob_col_idx['DueTime']]<=self.env_time)]
        # Sort by due date
        ots = ots[np.argsort(ots[:,self.ob_col_idx['DueTime']])]
        # Ship by product
        if len(ots) > 0:
            for p, inv in zip(self.product_ids, self.inventory):
                idx = self.prod_inv_idx[p]
                ots_prods = ots[np.where(ots[:,self.ob_col_idx['ProductID']].astype(int)==int(p))]
                cum_vol = ots_prods[:,self.ob_col_idx['Quantity']].cumsum()
                shipped_order_idx = np.where(cum_vol <= inv)[0]
                if len(shipped_order_idx) > 0:
                    self.inventory[idx] -= cum_vol[shipped_order_idx[-1]]                
                # Get shipped and late order numbers
                if shipped_orders is None:
                    shipped_orders = ots_prods[
                        shipped_order_idx,self.ob_col_idx['Number']]
                else:
                    shipped_orders = np.hstack([shipped_orders,
                        ots_prods[shipped_order_idx,self.ob_col_idx['Number']]])
                # Get late orders by keeping remaining OTS document numbers
                # TODO: if ship time != 24, then the current setup punishes
                # orders if they ship same day, but at a later time of day, 
                # e.g. shipment due at 17hrs, check at 24 hrs -> order late.
                if late_orders is None:
                    late_orders = np.delete(ots_prods[:,
                        self.ob_col_idx['Number']], shipped_order_idx)
                else:
                    late_orders = np.hstack([late_orders,
                        np.delete(ots_prods[:, self.ob_col_idx['Number']],
                            shipped_order_idx)])
        # Mark order as shipped, late, and on_time
        mark_shipped = np.isin(self.order_book[:,self.ob_col_idx['Number']], 
            shipped_orders)
        self.order_book[mark_shipped, self.ob_col_idx['Shipped']] = 1
        self.order_book[mark_shipped, self.ob_col_idx['ShipTime']] = self.env_time
        mark_late = np.isin(self.order_book[:,self.ob_col_idx['Number']],
            late_orders)
        self.order_book[mark_late, self.ob_col_idx['OnTime']] = -1
        # Mark on time if shipped and ActualGITime <= DueTime
        # Subset by current period shipments to avoid changing prev orders
        shipped_orders = self.order_book[mark_shipped]
        shipped_agit = shipped_orders[:, self.ob_col_idx['ShipTime']]
        shipped_pgit = shipped_orders[:, self.ob_col_idx['DueTime']]
        shipped_on_time_nums = shipped_orders[np.less_equal(
            shipped_agit, shipped_pgit), self.ob_col_idx['Number']]
        self.order_book[np.isin(self.order_book[:, self.ob_col_idx['Number']], 
            shipped_on_time_nums), self.ob_col_idx['OnTime']] = 1
        self._revenue += self._calc_revenue(mark_shipped, mark_late)
        self._late_penalties += self._calc_late_penalties()
        # Append shipment volumes
        agg_shipped_array = np.zeros(self.tot_products)
        unique_pid, unique_id = np.unique(
            shipped_orders[:, self.ob_col_idx['ProductID']], return_inverse=True)
        if len(unique_pid) > 0:
            agg_shipped = np.bincount(unique_id, shipped_orders[:, self.ob_col_idx['Quantity']])
            agg_shipped_array[self._vmap_pid_inventory(unique_pid)] += agg_shipped
        # self.containers.quantity_shipped.append(agg_shipped_array)

    def _calc_reward(self):
        reward = self._revenue - self._late_penalties - self._holding_costs
        self._revenue, self._late_penalties, self._holding_costs = 0, 0, 0
        return reward

    def _calc_revenue(self, shipped_orders, late_orders):
        shipped_this_period = self.order_book[shipped_orders]
        revenue = np.dot(
            shipped_this_period[:, self.ob_col_idx['Quantity']],
            shipped_this_period[:, self.ob_col_idx['Value']]
        )
        return revenue

    def _calc_holding_cost(self):
        holding_costs = np.dot(self.inventory, self._holding_cost_array)
        return holding_costs

    def _calc_late_penalties(self):
        un_shipped_late = self.order_book[np.where(
            (self.order_book[:, self.ob_col_idx['Shipped']]==0) &
            (self.order_book[:, self.ob_col_idx['OnTime']]<0)
        )]
        late_penalties = np.dot(
            un_shipped_late[:, self.ob_col_idx['Quantity']],
            un_shipped_late[:, self.ob_col_idx['Value']]
        ) * self.late_penalty
        return late_penalties

    def _get_state(self):
        '''
        Base method that returns inventory, order book, times, and schedules,
        as a dictionary for further processing.
        '''
        state_dict = {}
        state_dict = {'env_time': self.env_time}
        state_dict['schedules'] = {k: v.copy() for k, v in self.schedules.items()}
        state_dict['order_book'] = self.order_book[np.where(
            self.order_book[:, self.ob_col_idx['CreateTime']]<=self.env_time)[0]]
        state_dict['inventory'] = self.inventory.copy()

        return state_dict

    def __map_pid_to_inventory(self, prod_id):
        return self.prod_inv_idx[prod_id]

    def _vectorize_mappings(self):
        self._vmap_pid_inventory = np.vectorize(self.__map_pid_to_inventory)  

    def _initialize_demand_model(self):
        # Only initialize at beginning of model
        if self._init_demand:
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

    def _get_next_production_starts(self):
        '''
        Returns a deque of starting times to push through each stage.
        '''
        production_starts =[]
        for stage, _line in self._stage_line_dict.items():
            for line in _line.keys():
                schedule = self.schedules[stage][line].copy()
                if schedule is None:
                    continue
                current_batch = self.current_production[stage][line].Number
                next_prod = schedule[np.where(
                    (schedule[:, self.sched_col_idx['Completed']]==0) &
                    (schedule[:, self.sched_col_idx['ProdStartTime']]>=self.env_time) &
                    (schedule[:, self.sched_col_idx['Number']]!=current_batch))][0]
                production_starts.append(utils.ProdTuple(
                    Stage=stage,
                    Line=line,
                    Number=next_prod[self.sched_col_idx['Number']],
                    ProdStartTime=next_prod[self.sched_col_idx['ProdStartTime']],
                    ProdReleaseTime=next_prod[self.sched_col_idx['ProdReleaseTime']], 
                    ProductID=next_prod[self.sched_col_idx['ProductID']],
                    Quantity=next_prod[self.sched_col_idx['Quantity']]))
        # Return deque ordered by start time
        return deque(sorted(production_starts, key=attrgetter('ProdStartTime')))


    def _run_demand_model(self):
        '''
        Base model calculates mean run rate for the products, multiplies
        this by the number of hours in the simulation, and uses this as
        the mean, total demand for the each episode. A fraction of the
        total demand is then split among the various final products. Time
        series are made by sampling orders from a normalized sine wave to
        simulate seasonality. Random values (e.g. demand shares) are
        fixed whenever or_gym.make() is called, and are preserved during each
        call to reset().
        Returns order_book object containing demand data for the episode.
        '''
        self._initialize_demand_model()
        order_book = np.zeros((self.product_demand.sum(), 
            len(self.order_book_cols)))
        order_book[:, self.ob_col_idx['Number']] = np.arange(
            self.product_demand.sum())
        prods = np.hstack([np.repeat(i, j)
            for i, j in zip(self.product_ids, self.product_demand)])
        order_book[:, self.ob_col_idx['ProductID']] = prods
        due_dates = np.hstack([np.random.choice(np.arange(0, self.simulation_days),
            p=self._p_seas, size=i)
            for i in self.product_demand])
        order_book[:, self.ob_col_idx['DueTime']] = due_dates
        order_book[:, self.ob_col_idx['CreateTime']] = due_dates - \
            np.random.poisson(lam=self.avg_lead_time, 
                size=self.product_demand.sum())

        return order_book

    def _maintentance_model(self):
        return False

    def run_maintentance_model(self):
        return self._maintentance_model()

    def _book_inventory(self, prod_tuple):
        # Moves completed production into inventory.
        prod_idx = self.prod_inv_idx[prod_tuple.ProductID]
        self.inventory[prod_idx] += prod_tuple.Quantity
        self._mark_booked_in_schedule(prod_tuple)

    def _mark_booked_in_schedule(self, prod_tuple):
        # Changes booked column from 0 to 1
        sched = self.schedules[prod_tuple.Stage][prod_tuple.Line]
        row_to_book = np.where(
            sched[:, self.sched_col_idx['Number']]==prod_tuple.Number)
        sched[row_to_book, self.sched_col_idx['Completed']] = 1
        sched[row_to_book, self.sched_col_idx['Quantity']] = prod_tuple.Quantity
        self.schedules[prod_tuple.Stage][prod_tuple.Line] = sched.copy()

    def append_schedules(self, action):
        # Sampling passes actions as OrderedDict
        # if type(action) is OrderedDict:
            # action = action.values()
        # Ray passes actions as dictionary values
        if type(action) is dict or type(action) is OrderedDict:
            for stage, _l in action.items():
                for line, act in _l.items():
                    schedule = self.schedules[stage][line]
                    self._append_schedule(stage, line, act, schedule)
        else:
            # If actions passed as array, it must be ordered correctly
            c = 0
            for stage, _l in self._stage_line_dict.items():
                for line in _l.keys():
                    schedule = self.schedules[stage][line]
                    act = action[c]
                    self._append_schedule(stage, line, act, schedule)
                    c += 1
            
    def _append_schedule(self, stage, line, action, schedule):
        # Get values from mappings
        prod_id = self.get_prod_id_from_action(stage, line, action)
        qty = self.get_prod_qty(stage, line, prod_id)
        completed = 0
        if schedule is None:
            num = 1
            start_time = self.env_time
            last_prod_id = prod_id # Assume no transition at start
        else:
            last_prod_id = schedule[-1, self.sched_col_idx['ProductID']]
            if prod_id == last_prod_id:
                # TODO: Extend current entry by some discrete amount
                # be it minimum batch size or some other value.
                pass
            num = schedule[-1, self.sched_col_idx['Number']] + 1
            start_time = schedule[-1, self.sched_col_idx['ProdEndTime']]  
        
        off_grade_hrs = self.get_off_grade_time(stage, line, last_prod_id, prod_id)
        prod_rate = self.get_production_rate(stage, line, prod_id)
        off_grade = off_grade_hrs * prod_rate
        end_time = np.ceil(start_time + off_grade_hrs + qty / prod_rate)
        release_time = end_time + self.get_cure_time(stage, line, prod_id)
        
        new_entry = np.zeros(len(self.sched_col_idx)) # Next line in schedule
        # Add values to new_entry
        new_entry[self.sched_col_idx['Stage']] = stage
        new_entry[self.sched_col_idx['Line']] = line
        new_entry[self.sched_col_idx['Number']] = num
        new_entry[self.sched_col_idx['ProductID']] = prod_id
        new_entry[self.sched_col_idx['Quantity']] = qty
        new_entry[self.sched_col_idx['ProdStartTime']] = start_time
        new_entry[self.sched_col_idx['ProdEndTime']] = end_time
        new_entry[self.sched_col_idx['ProdReleaseTime']] = release_time
        new_entry[self.sched_col_idx['OffGrade']] = off_grade
        new_entry[self.sched_col_idx['Completed']] = completed
        
        if schedule is None:
            self.schedules[stage][line] = new_entry.reshape(1, -1).copy().astype(int)
        else:
            self.schedules[stage][line] = np.vstack([schedule, new_entry.astype(int)])

    #########################################################################
    # The following functions are set up to be ammenable for multi-stage/multi-line
    # environments, so many of the arguments currently go unused.
    # Will make more consistent in the future.
    #########################################################################
    def get_prod_qty(self, stage, line, prod_id):
        return self.min_product_qtys[prod_id]

    def get_prod_id_from_action(self, stage, line, action):
        idx = np.where(action==self._stage_line_dict[stage][line])[0].take(0)
        return self.product_ids[idx]

    def get_off_grade_time(self, stage, line, prod_from, prod_to):
        return self.transition_dict[(prod_from, prod_to)]

    def get_production_rate(self, stage, line, prod_id):
        return self.run_rates[prod_id]

    def get_cure_time(self, stage, line, prod_id):
        return self.cure_times[prod_id]

    def get_demand(self):
        return self._run_demand_model()

    def get_state(self):
        return self._get_state()

    def _RESET(self):
        # Deque of products to release
        self.production_deque = deque()
        self.production_release_deque = deque()
        self.inventory = self.init_inventory.copy()
        self.env_time = 0
        self._revenue = 0
        self._holding_cost = 0
        self._late_penalties = 0
        self.order_book = self.get_demand()
        self.schedules = self._init_schedules()
        self.current_production = self._init_current_production()
        return self.get_state()

    #########################################################################
    # Abstract methods must be defined by the child class, otherwise an error
    # will be raised.
    #########################################################################
    @abstractmethod
    def step(self, action):
        raise NotImplementedError("step() method not implemented.")

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset() method not implemented.")

    @abstractmethod
    def _init_schedules(self):
        raise NotImplementedError("_init_schedules method not implemented.")

    @abstractmethod
    def _init_stage_line_dict(self):
        raise NotImplementedError("_init_stage_line_dict method not implemented.")

    @abstractmethod
    def _init_action_inputs(self):
        raise NotImplementedError("_init_action_inputs method not implemented.")

    @abstractmethod
    def _init_current_production(self):
        raise NotImplementedError("_init_current_production method not implemented.")

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
        utils.assign_env_config(self, kwargs)
        super().__init__()

        self.action_space = spaces.Dict({k:
            spaces.Dict({k1: spaces.Discrete(len(v1))
                for k1, v1 in v.items()})
            for k, v in self._stage_line_dict.items()})

        self.observation_space = None
        
        self.__init_prod = utils.ProdTuple(None, None, None, None, None, None, None)
        self._action_inputs = self._init_action_inputs()
        self.reset()

    def _init_stage_line_dict(self):
        # Nested dictionary to link products to stages and lines
        return {0: {0: self.product_ids}}

    def _init_schedules(self):
        '''Returns nested dictionary with empty schedules.'''
        return {k: {k1: None for k1 in v.keys()} 
            for k, v in self._stage_line_dict.items()}

    def _init_action_inputs(self):
        '''Returns dimensionality of action space.'''
        return len([j for i in self.action_space 
            for j in self.action_space[i]])

    def _init_current_production(self):
        return {stage:
            {line: self.__init_prod for line in v.keys()} 
                for stage, v in self._stage_line_dict.items()}

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()