#!usr/bin/env python

import gym
import or_gym
from pyomo.environ import *

def build_ukp_ip_model(env):
    assert env.spec.id == 'Knapsack-v0', \
        '{} received. Heuristic designed for Knapsack-v0.'.format(env.spec.id)
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
    m.x = Var(m.i, within=NonNegativeIntegers)

    @m.Constraint()
    def weight_constraint(m):
        return sum(m.w[i] * m.x[i] for i in m.i) - m.W <= 0

    m.obj = Objective(expr=(
        sum([m.v[i] * m.x[i] for i in m.i])),
        sense=maximize)

    return m

def build_bkp_ip_model(env):
    assert env.spec.id == 'Knapsack-v0', \
        '{} received. Heuristic designed for Knapsack-v0.'.format(env.spec.id)
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

def build_okp_ip_model(env):
    assert env.spec.id == 'Knapsack-v0', \
        '{} received. Heuristic designed for Knapsack-v0.'.format(env.spec.id)
    env.reset()
    pass