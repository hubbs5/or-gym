#!usr/bin/env python

import gym
import or_gym
from pyomo.environ import *
import numpy as np

def build_ukp_ip_model(env):
    assert env.spec.id == 'Knapsack-v0', \
        '{} received. Heuristic designed for Knapsack-v0.'.format(env.spec.id)

    # Initialize model
    m = ConcreteModel()

    # Sets, parameters, and variables
    m.W = env.max_weight
    m.i = Set(initialize=env.item_numbers)
    m.w = Param(m.i, 
        initialize={i: j for i, j in zip(env.item_numbers, env.item_weights)})
    m.v = Param(m.i, 
        initialize={i: j for i, j in zip(env.item_numbers, env.item_values)})
    m.x = Var(m.i, within=NonNegativeIntegers)

    @m.Constraint()
    def weight_constraint(m):
        return sum(m.w[i] * m.x[i] for i in m.i) - m.W <= 0

    m.obj = Objective(expr=(
        sum([m.v[i] * m.x[i] for i in m.i])),
        sense=maximize)

    return m

def build_bin_ip_model(env):
    assert env.spec.id == 'Knapsack-v1', \
        '{} received. Heuristic designed for Knapsack-v1.'.format(env.spec.id)

    # Initialize model
    m = ConcreteModel()

    # Sets, parameters, and variables
    m.W = env.max_weight
    m.i = Set(initialize=env.item_numbers)
    m.w = Param(m.i, 
        initialize={i: j for i, j in zip(env.item_numbers, env.item_weights)})
    m.v = Param(m.i, 
        initialize={i: j for i, j in zip(env.item_numbers, env.item_values)})
    m.x = Var(m.i, within=Binary)

    @m.Constraint()
    def weight_constraint(m):
        return sum(m.w[i] * m.x[i] for i in m.i) - m.W <= 0

    m.obj = Objective(expr=(
        sum([m.v[i] * m.x[i] for i in m.i])),
        sense=maximize)

    return m

def build_bkp_ip_model(env):
    assert env.spec.id == 'Knapsack-v2', \
        '{} received. Heuristic designed for Knapsack-v2.'.format(env.spec.id)
    env.reset()

    # Initialize model
    m = ConcreteModel()

    # Sets, parameters, and variables
    m.W = env.max_weight
    m.i = Set(initialize=env.item_numbers)
    m.w = Param(m.i, 
        initialize={i: j for i, j in zip(env.item_numbers, env.item_weights)})
    m.v = Param(m.i, 
        initialize={i: j for i, j in zip(env.item_numbers, env.item_values)})
    m.b = Param(m.i,
        initialize={i: j for i, j in zip(env.item_numbers, env.item_limits)})
    m.x = Var(m.i, within=NonNegativeIntegers)

    @m.Constraint()
    def weight_constraint(m):
        return sum(m.w[i] * m.x[i] for i in m.i) - m.W <= 0

    @m.Constraint(m.i)
    def item_constraint(m, i):
        return m.x[i] - m.b[i] <= 0

    m.obj = Objective(expr=(
        sum([m.v[i] * m.x[i] for i in m.i])),
        sense=maximize)

    return m

def build_okp_ip_model(env, scenario=None):
    '''This model returns the optimal solution.'''
    assert env.spec.id == 'Knapsack-v3', \
        '{} received. Heuristic designed for Knapsack-v3.'.format(env.spec.id)
    env.reset()
    if scenario is None:
        scenario = np.random.choice(env.item_numbers, 
            p=env.item_probs, size=env.step_limit)
    
    # Selected items
    ordered_weights = [env.item_weights[i] for i in scenario]
    ordered_values = [env.item_values[i] for i in scenario]
    # Initialize model
    m = ConcreteModel()

    # Sets, parameters, and variables
    m.W = env.max_weight
    m.T = env.step_limit

    m.i = Set(initialize=np.arange(m.T))

    m.w = Param(m.i, 
        initialize={i: j for i, j in zip(
            np.arange(m.T), ordered_weights)})
    m.v = Param(m.i, 
        initialize={i: j for i, j in zip(
            np.arange(m.T), ordered_values)})

    m.x = Var(m.i, within=Binary)

    @m.Constraint()
    def weight_constraint(m):
        return sum(m.w[i] * m.x[i] 
                   for i in m.i) - m.W <= 0
        
    @m.Constraint()
    def selection_constraint(m):
        return sum(m.x[i] for i in m.i) - m.T <= 0

    m.obj = Objective(expr=(
        sum([m.v[i] * m.x[i] for i in m.i])),
        sense=maximize)

    return m