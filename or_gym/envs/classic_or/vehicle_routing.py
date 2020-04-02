import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
import copy

class VehicleRoutingEnv(gym.Env):
    '''
    The Vehicle Routing Problem is a classic.....

    In this version of the problem, there are 3 cars vehicles that receive pick-up 
    orders dynamically on a 10 x 10 grid.
    Each grid spot is numbered starting 0 to 99, left to right, up to bottom. 
    The size of the order to be picked-up is variable and also revealed dynamically.
    Each region in the grid is numbered and modeled as 15 minutes apart. 
    The cars begin at the depot which is in spot self.depot_location. 
    Each step involves each car moving 1 spot in the grid (or not moving), 15 min elapsing, 
    and chosing to pick up or leave the item. 
    If >1 vehicles are in the same spot (and there is demand) and >1 has chosen to pick it up, 
    it will go the first vehicle in order 1-2-3 that has capacity to accept it. 
    Each vehicle has a fixed capacity that is reset upon every visit to the depot.
    Each vehicle has 8 hours to collect as much reward before (ideally) finishing the day at 
    the depot.  
    
    Observation:
        Type: Box(107) 
        "vehicle locations" (idx 0 1 2): location 0-99 of vehicles (v) [v1, v2, v3]
        "vehicle load" (idx 3 4 5): current load (l) 0-C of vehicles [l1, l2, l3]
        "time period" (idx 6): current 15 minute time period in workday
        "amount demand" (idx 7-106): amount of pick-up demand in each point in grid 

    Action: 
        Type: Box(3) 
        "vehicle move direction" (idx 0 1 2): 0 - No movement; 1 - Up; 2 - Down; 3 - Left; 4 - Right;
                                For vehicles (v) [v1, v2, v3]
        "vehicle pickup order" (idx 3, 4, 5): 0 - deny order; 1 - pickup order; for vehicles (v) [v1, v2, v3]

    Reward: 
        If order accepted keeps vehicle below/at capacity, the reward is the positive size of the order. Else, 
        the reward is the negative of the size of the order accepted. Movement incurs a negative reward. 
        Failing to end the day at the depot incurs a negative reward based on distance to the depot. 
        A negative reward is incurred for any unfulfilled demand outstanding (negative of the sum).

    Starting State: 
        All vehicles are located in spot self.depot_location, with an empty load, at the beginning of time period 0, with an initial set of orders
        (i.e. non-zero demand). 

    Episode Terimantion: 
        Episode termination occurs when all time periods for the day have elapsed. 

    '''

    def __init__(self, *args, **kwargs):
        # Immutable parameters
        self.vehicle_max_capacity = 50
        self.movement_cost  = 0.2
        self.after_hours_movement_multipler = 2
        self.max_time_period = 31 # Should this be 32?
        self.num_vehicles = 3
        self.num_directions = 5 # None, NSEW
        self.num_actions = 2 # Pickup or nothing
        self.depot_location = 56
        self.num_locs = 100
        self._max_reward = 500
        # Add env_config, if any
        for key, value in kwargs.items():
            setattr(self, key, value)
        if hasattr(self, 'env_config'):
            for key, value in self.env_config.items():
                setattr(self, key, value)

        # State and action space definitions
        self.observation_space = spaces.Box(0, self.num_locs,
        	shape=(self.num_locs+2*self.num_vehicles + 1,))
        self.action_space = spaces.MultiDiscrete([
        	self.num_directions if i < self.num_vehicles
        	else self.num_actions for i in range(2*self.num_vehicles)])
        # self.action_space = spaces.Tuple(
        # 	[spaces.Discrete(self.num_directions) if i < self.num_vehicles
        #     else spaces.Discrete(self.num_actions)
        #     for i in range(2*self.num_vehicles)])
        self.seed()
        self.reset()

    def step(self, action):
        reward = 0
        # Movement actions ('if' statement ensures movement off grid does not occur)
        for idx, a in enumerate(action[:self.num_vehicles]):
            # Move up
            if a == 1:
                if self.vehicle_locations[idx] > 9:
                    self.vehicle_locations[idx] -= 10
                    reward -= self.movement_cost
            # Down
            if a == 2:
                if self.vehicle_locations[idx] < 90:
                    self.vehicle_locations[idx] += 10
                    reward -= self.movement_cost
            # Left
            if a == 3:
                if self.vehicle_locations[idx] not in np.linspace(0,90,10):
                    self.vehicle_locations[idx] -= 1
                    reward -= self.movement_cost
            # Right
            if a == 4:
                if self.vehicle_locations[idx] not in np.linspace(9,99,10):
                    self.vehicle_locations[idx] += 1
                    reward -= self.movement_cost

        # Item loading (demand statisfaction) actions
        for idx, a in enumerate(action[self.num_vehicles:2*self.num_vehicles]):
            # If car chooses to pick up item, make sure it fits; if so, demand is set to 0  
            if a == 1:
                if self.demand[self.vehicle_locations[idx]] + \
                    self.vehicle_load[idx] <= self.vehicle_max_capacity:
                    
                    self.vehicle_load[idx] += self.demand[self.vehicle_locations[idx]]
                    reward += self.demand[self.vehicle_locations[idx]]
                    self.demand[self.vehicle_locations[idx]] = 0
            if self.vehicle_locations[idx] == self.depot_location:
                self.vehicle_load[idx] = 0

        self.time_period += 1
        self._update_state()

        if self.time_period > self.max_time_period: 
            done = True
            for location in self.vehicle_locations:
                reward -= self.distance_from_depot(location) * \
                    self.movement_cost * self.after_hours_movement_mult
            reward -= np.sum(self.demand)
        else:
            self.demand = self.update_demand()
            done = False

        return self.state, reward, done, {}

    def reset(self):
        self.vehicle_locations = np.repeat(self.depot_location, self.num_vehicles)
        self.vehicle_load = np.zeros(self.num_vehicles)
        self.time_period = 0
        self.demand = self.generate_initial_demand()
        self._update_state()
        return self.state

    def _update_state(self):
        self.state = np.concatenate([
        	self.vehicle_locations,
            self.vehicle_load,
            np.array([self.time_period]),
            self.demand])

    def sample_action(self):
    	return self.action_space.sample()
    	
    def _get_obs(self):
        return self.state

    def generate_initial_demand(self):
        demand = np.zeros(self.num_locs)
        # Pick 10 random points in grid (excluding depot) to have random
        # demand according to "random" choice of normal distribution
        for point in np.random.choice(
            np.setdiff1d(np.arange(self.num_locs),
                self.depot_location), size=10):
            demand[point] = max(0, np.random.normal(
                np.random.choice([4, 8, 10]),
                np.random.choice([1, 2, 3])))
        return demand

    def update_demand(self):
        if self.time_period in range(1,12):
            prob_new_order = 0.25
        elif self.time_period in range(12, 24):
            prob_new_order = 0.15
        else:
            prob_new_order = 0.10

        if np.random.choice([1,0],
            p=[prob_new_order, 1-prob_new_order]) == 1:
            for point in np.random.choice(
                np.setdiff1d(np.arange(self.num_locs),
                    self.depot_location),
                size=np.random.choice([1,2])):
                self.demand[point] += max(0, np.random.normal(
                    np.random.choice([4, 8, 10]),
                    np.random.choice([1, 2, 3])))

        return self.demand

    def distance_from_depot(self, location):
        depot_x = self.depot_location//10
        depot_y = self.depot_location%10
        location_x = location//10
        location_y = location%10
        return abs(location_x-depot_x) + abs(location_y-depot_y)