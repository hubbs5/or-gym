'''
Example taken from Balaji et al.
Paper: https://arxiv.org/abs/1911.10641
GitHub: https://github.com/awslabs/or-rl-benchmarks
'''
import gym
from gym import spaces
import or_gym
from or_gym.utils import assign_env_config
import random
import numpy as np
from scipy.stats import truncnorm


class VehicleRoutingEnv(gym.Env):
    '''
    Dynamic Vehicle Routing Problem

    This environment simulates a driver working with a food delivery app
    to move through a city, accept orders, pick them up from restaurants,
    and deliver them to waiting customers. Each order has a specific
    delivery value, restaurant, and delivery location, all of which are 
    known by the driver before he accepts the order. After accepting, the
    driver must navigate to the restaurant to collect the order and then
    deliver it. If an order isn't accepted, it may be taken by another
    driver. Additionally, the driver has 60 minutes to make a delivery
    from the time an order is created. 
    The city is represented as a grid with different zones that have
    different statistics for order creation and value. At each time step,
    new orders are created with a fixed probability unique to each zone.
    The driver's vehicle also has a finite capacity limiting the number of
    orders he can carry at a given time, although there is no limit on the
    number of accepted orders.
    The driver receives a penalty for time and distance spent during travel,
    but receives rewards for accepting and delivering orders.

    Observation:
        Type: Box
        State Vector: S = (p, h, c, l, w, e, v)
        p = pickup location
        h = driver's current position
        c = remaining vehicle capacity
        l = order location
        w = order status (open, accepted, picked up, delivered/inactive)
        e = time elapsed since order generation
        v = order value

    Action: 
        Type: Discrete
        0 = wait
        1:max_orders = accept order
        max_orders:2*max_orders = pickup order
        2*max_orders:3*max_orders = deliver order
        3*max_orders:3*max_orders + n_restaurants = go to restaurant

        Action masking is available for this environment. Set mask=True
        in the env_config dictionary.

    Reward:
        The agent recieves 1/3 of the order value for accepting an order,
        picking it up, and delivering the order. The cost is comprised of
        three elements: delivery time, delivery distance, and cost of failure
        (if the driver does not deliver the item). 

    Starting State:
        Restaurant and driver locations are randomized at the start of each
        episode. New orders are generated according to the order probability.

    Episode Terimantion:
        Episode termination occurs when the total time has elapsed.
    '''

    def __init__(self, *args, **kwargs):
        self.n_restaurants = 2
        self.max_orders = 10
        self.order_prob = 0.5
        self.vehicle_capacity = 4
        self.grid = (5, 5)
        self.order_promise = 60
        self.order_timeout_prob = 0.15
        self.num_zones = 4
        self.order_probs_per_zone = [0.1, 0.5, 0.3, 0.1]
        self.order_reward_min = [8, 5, 2, 1]
        self.order_reward_max = [12, 8, 5, 3]
        self.half_norm_scale_reward_per_zone = [0.5, 0.5, 0.5, 0.5]
        self.penalty_per_timestep = 0.1
        self.penalty_per_move = 0.1
        self.order_miss_penalty = 50
        self.step_limit = 1000
        self.mask = False
        self.info = {}

        assign_env_config(self, kwargs)
        self._order_nums = np.arange(self.max_orders)
        self.loc_permutations = [(x, y) for x in range(self.grid[0])
                                 for y in range(self.grid[1])]
        self.action_dim = 1 + 3 * self.max_orders + self.n_restaurants
        self.obs_dim = 2 * self.n_restaurants + 4 + 6 * self.max_orders
        box_low = np.zeros(self.obs_dim)
        box_high = np.hstack([
            np.repeat(
                max(self.grid), 2 * self.n_restaurants + 2),  # Locations 0-5
            np.repeat(self.vehicle_capacity, 2),  # Vehicle capacities 6-7
            np.tile(np.hstack([4, self.n_restaurants, self.grid,
                               self.order_promise, max(self.order_reward_max)]), self.max_orders)
        ])

        if self.mask:
            self.observation_space = spaces.Dict({
                'action_mask': spaces.Box(
                    low=np.zeros(self.action_dim),
                    high=np.ones(self.action_dim),
                    dtype=np.uint8),
                'avail_actions': spaces.Box(
                    low=np.zeros(self.action_dim),
                    high=np.ones(self.action_dim),
                    dtype=np.uint8),
                'state': spaces.Box(
                    low=box_low,
                    high=box_high,
                    dtype=np.float16)
            })
        else:
            self.observation_space = spaces.Box(
                low=box_low,
                high=box_high,
                dtype=np.float16)

        self.action_space = spaces.Discrete(self.action_dim)

        self.reset()

    def _STEP(self, action):
        done = False
        self.reward = 0
        self.late_penalty = 0

        if action == 0:
            self.wait(action)
        elif action <= self.max_orders:
            self.accept_order(action)
        elif action <= 2 * self.max_orders:
            self.pickup_order(action)
        elif action <= 3 * self.max_orders:
            self.deliver_order(action)
        elif action <= 3 * self.max_orders + self.n_restaurants:
            self.return_to_restaurant(action)
        else:
            raise Exception(
                f"Selected action ({action}) outside of action space.")

        self.state = self._update_state()

        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True

        return self.state, self.reward, done, self.info

    def wait(self, action):
        # Do nothing
        pass

    def accept_order(self, action):
        # Accept order denoted by action
        order_idx = action - 1
        if order_idx not in self.order_dict.keys():
            # Invalid action, do nothing
            pass
        elif self.order_dict[order_idx]['Status'] == 1:
            self.order_dict[order_idx]['Status'] = 2
            self.reward += self.order_dict[order_idx]['Value'] / 3

    def pickup_order(self, action):
        order_idx = action - self.max_orders - 1
        if order_idx not in self.order_dict.keys():
            # Invalid action, do nothing
            pass
        else:
            restaurant = self.order_dict[order_idx]['RestaurantID']
            restaurant_loc = self.restaurant_loc[restaurant]
            self._go_to_destination(restaurant_loc)
            self.reward -= self.penalty_per_move
            # Movement and pickup can occur during same time step
            if self.order_dict[order_idx]['Status'] == 2 and self.driver_loc[0] == restaurant_loc[0] and self.driver_loc[1] == restaurant_loc[1]:
                if self.vehicle_load < self.vehicle_capacity:
                    self.order_dict[order_idx]['Status'] = 3
                    self.vehicle_load += 1
                    self.reward += self.order_dict[order_idx]['Value'] / 3

    def deliver_order(self, action):
        order_idx = action - 2 * self.max_orders - 1
        if order_idx not in self.order_dict.keys():
            # Invalid action, do nothing
            pass
        else:
            order_loc = self.order_dict[order_idx]['DeliveryLoc']
            self._go_to_destination(order_loc)
            self.reward -= self.penalty_per_move
            # Can deliver multiple orders simultaneously
            for k, v in self.order_dict.items():
                if v['Status'] == 3 and v['DeliveryLoc'][0] == self.driver_loc[0] and v['DeliveryLoc'][1] == self.driver_loc[1]:
                    if v['Time'] <= self.order_promise:
                        self.reward = v['Value'] / 3
                    self.vehicle_load -= 1
                    v['Status'] = 4  # Delivered

    def return_to_restaurant(self, action):
        restaurant = action - 3 * self.max_orders - 1
        restaurant_loc = self.restaurant_loc[restaurant]
        self._go_to_destination(restaurant_loc)
        self.reward -= self.penalty_per_move

    def _update_orders(self):
        self._update_order_times()
        self._remove_orders()
        self._generate_orders()

    def _remove_orders(self):
        # Remove orders if they're over due
        orders_to_delete = []
        for k, v in self.order_dict.items():
            if v['Time'] >= self.order_promise:
                if v['Status'] >= 2:
                    # Apply penalty and remove associated rewards
                    self.reward -= (self.order_miss_penalty +
                                    v['Value'] * (v['Status'] == 2)/3 +
                                    v['Value'] * (v['Status'] == 3) * 2/3)
                    self.late_penalty += self.order_miss_penalty
                if v['Status'] == 3:
                    self.vehicle_capacity -= 1
                orders_to_delete.append(k)

            elif v['Status'] == 4:
                orders_to_delete.append(k)

            # Probabalistically remove open orders
            elif v['Status'] == 1 and np.random.random() < self.order_timeout_prob:
                orders_to_delete.append(k)

        for k in orders_to_delete:
            del self.order_dict[k]

    def _update_state(self):
        self._update_orders()
        # Placeholder for order data
        order_array = np.zeros((self.max_orders, 6))
        try:
            order_data = np.hstack([v1 for v in self.order_dict.values()
                                    for v1 in v.values()]).reshape(-1, 7)
            order_array[order_data[:, 0].astype(int)] += order_data[:, 1:]
        except ValueError:
            # Occurs when order_data is empty
            pass
        state = np.hstack([
            np.hstack(self.restaurant_loc),
            np.hstack(self.driver_loc),
            np.hstack([self.vehicle_load, self.vehicle_capacity]),
            order_array.flatten()
        ])
        if self.mask:
            action_mask = self._update_mask(state)
            state = {
                'state': state,
                'action_mask': action_mask,
                'avail_actions': np.ones(self.action_dim)
            }
        return state

    def _update_mask(self, state):
        action_mask = np.zeros(self.action_dim)
        # Wait and return to restaurant are always allowed
        action_mask[0] = 1
        action_mask[(3 * self.max_orders + 1)                    :(3 * self.max_orders + self.n_restaurants + 1)] = 1

        for k, v in self.order_dict.items():
            status = v['Status']
            # Allow accepting an open order
            if status == 1:
                action_mask[k + 1] = 1
            # Allow navigating to accepted order for pickup
            elif status == 2 and self.vehicle_load < self.vehicle_capacity:
                action_mask[k + self.max_orders + 1] = 1
            # Allow delivery of picked up order
            elif status == 3:
                action_mask[k + 2 * self.max_orders + 1] = 1

        return action_mask

    def _RESET(self):
        self.step_count = 0
        self.vehicle_load = 0
        self.randomize_locations()
        self.zone_loc = self._get_zones()
        self.order_dict = {}
        self.state = self._update_state()
        return self.state

    def _update_order_times(self):
        for k, v in self.order_dict.items():
            if v['Status'] >= 1:
                v['Time'] += 1

    def _generate_orders(self):
        open_slots = self._order_nums[~np.isin(self._order_nums,
                                               np.array([k for k in self.order_dict.keys()]))]
        try:
            order_num = open_slots.min()
        except ValueError:
            pass
        for n in open_slots:
            # Probabalistically create a new order
            if np.random.random() < self.order_prob:
                zone = np.random.choice(
                    self.num_zones, p=self.order_probs_per_zone)
                order = self._get_order_from_zone(zone, order_num)
                self.order_dict[order_num] = order
                order_num += 1

    def _get_order_from_zone(self, zone, n):
        delivery_loc = random.choice(self.zone_loc[zone])
        restaurant_idx = np.random.choice(self.n_restaurants)
        value = truncnorm.rvs(0,
                              (self.order_reward_max[zone] -
                               self.order_reward_min[zone])
                              / self.half_norm_scale_reward_per_zone[zone],
                              self.order_reward_min[zone],
                              self.half_norm_scale_reward_per_zone[zone])
        return {'Number': n,
                'Status': 1,
                'RestaurantID': restaurant_idx,
                'DeliveryLoc': delivery_loc,
                'Time': 0,
                'Value': value}

    def randomize_locations(self):
        self._place_restaurants()
        self._place_driver()

    def _place_restaurants(self):
        self.restaurant_loc = random.sample(
            self.loc_permutations, self.n_restaurants)

    def _place_driver(self):
        self.driver_loc = list(random.sample(self.loc_permutations, 1)[0])

    def _move_driver(self, direction):
        if direction is None:
            return None
        # Receives direction from routing function
        if direction == 0:  # Up
            self.driver_loc[1] += 1
        elif direction == 1:  # Down
            self.driver_loc[1] -= 1
        elif direction == 2:  # Right
            self.driver_loc[0] += 1
        elif direction == 3:  # Left
            self.driver_loc[0] -= 1
        # Check boundaries
        if self.driver_loc[0] > self.grid[0]:
            self.driver_loc[0] = self.grid[0]
        if self.driver_loc[0] < 0:
            self.driver_loc[0] = 0
        if self.driver_loc[1] > self.grid[1]:
            self.driver_loc[1] = self.grid[1]
        if self.driver_loc[1] < 0:
            self.driver_loc[1] = 0

    def _go_to_destination(self, destination):
        # Automatically selects direction based on starting location and
        # destination.
        # 0 -> Up; 1 -> Down; 2 -> Right; 3 -> Left
        x_diff = self.driver_loc[0] - destination[0]
        y_diff = self.driver_loc[1] - destination[1]
        if abs(x_diff) >= abs(y_diff):
            if x_diff > 0:
                direction = 3
            elif x_diff < 0:
                direction = 2
            elif abs(x_diff) == abs(y_diff):  # 0 == 0
                # Do nothing
                direction = None
        else:
            if y_diff > 0:
                direction = 1
            elif y_diff < 0:
                direction = 0
        print('direction ',direction)
        self._move_driver(direction)

    def _get_num_spaces_per_zone(self):
        total_spaces = self.grid[0] * self.grid[1]
        spaces_per_zone = np.array([np.floor(total_spaces / self.num_zones)
                                    for i in range(self.num_zones)])
        for i in range(total_spaces % self.num_zones):
            spaces_per_zone[i] += 1
        return spaces_per_zone.astype(int)

    def _get_zones(self):
        # Slices the grid into zones by row
        spaces_per_zone = self._get_num_spaces_per_zone()
        zones = {}
        for i, n in enumerate(spaces_per_zone):
            x = sum(spaces_per_zone[:i])
            zones[i] = self.loc_permutations[x:x+n]

        zones = self._remove_restaurants_from_zone_locs(zones)
        return zones

    def _remove_restaurants_from_zone_locs(self, zones):
        for k, v in zones.items():
            for r in self.restaurant_loc:
                try:
                    loc_to_remove = v.index(r)
                    del zones[k][loc_to_remove]
                except ValueError:
                    pass
        return zones

    def step(self, action):
        return self._STEP(action)

    def reset(self):
        return self._RESET()
