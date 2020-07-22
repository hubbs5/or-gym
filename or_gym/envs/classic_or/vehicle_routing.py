'''
Example taken from Balaji et al.
Paper: https://arxiv.org/abs/1911.10641
GitHub: https://github.com/awslabs/or-rl-benchmarks
'''
import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils.env_config import assign_env_config
import copy

class VehicleRoutingEnv(gym.Env):
    '''
    Dynamic Vehicle Routing Problem

    This environment simulates a driver moving through a city to picke up
    orders as they are sent to him on a phone app. Each order has a specific
    delivery charge that is known by the driver and a pickup location that 
    the driver must navigate to in order to make the pickup. The city is 
    represented as a grid map with different zones for order creation. At
    each time step, new orders are created according to a fixed probability
    unique to each zone. Additionally, orders have delivery time windows that
    last for 60 minutes from order creation. Orders must be accepted prior to
    delivery and there is a fixed time-out probability if the driver does not
    accept it. The vehicle has a finite capacity of four orders, but there is
    no limit on the number of orders the driver can accept. The driver
    receives a penalty for each time step and distance traveled. The driver's
    goal is to maximize the reward represented by the orders minus the cost
    of the orders over the course of the episode.
    
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
        0 = accept open order
        1 = pick up the accepted order
        2 = go to the pick up location?
        3 = wait
        4 =

    Reward:
        The agent recieves 1/3 of the order value for accepting an order,
        picking it up, and delivering the order. The cost is comprised of
        three elements: delivery time, delivery distance, and cost of failure
        (if the driver does not deliver the item). 
    
    Starting State:

    Episode Terimantion:
        Episode termination occurs when the total time has elapsed.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass