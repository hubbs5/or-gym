from pyomo.environ import *
from pyomo.opt import SolverFactory
import time
import numpy as np

def solve_math_program(model, solver='glpk', print_results=True):

    solver = SolverFactory(solver)
    results = solver.solve(model, tee=print_results)
    return model, results

def solve_shrinking_horizon_mp(env, build_opt_model, action_func, 
    solver='glpk', print_results=True):
    '''
    Some models may be too large to solve to optimality, but we can 
    approximate an optimal, stochastic solution using a shrinking 
    horizon model. This optimizes the environment at each time step
    and carries the results forward to re-optimize at the next time step
    while working towards a fixed point in the future. Thus, each time
    horizon becomes smaller.

    Inputs:
        env: environment to optimize
        build__opt_model: function to build Pyomo optimization model
        action_func: function to extract actions from Pyomo model
        solver: string to specify math programming solver
        print_results: boolean to print updates at a given interval or not
    '''
    actions, rewards = [], []
    done = False
    count = 0
    t0 = time.time()
    while done == False:
        t1 = time.time()
        model = build_opt_model(env)
        model, results = solve_math_program(model, solver, print_results)
        action = action_func(model)
        s, r, done, info = env.step(action)
        actions.append(action)
        rewards.append(r)
        count += 1
        t2 = time.time()
        if print_results:
            if count % 10 == 0:
                print("Steps/s: {:.2f}\tTotal time (s): {:.2f}\t".format(
                    count, t2-t0))

    return model, actions, rewards

def extract_vm_packing_plan(model):
    plan = []
    for v in model.v:
        for t in model.t:
            if v == t:
                for n in model.n:
                    if model.x[n, v, t].value is None:
                        continue
                    if model.x[n, v, t].value > 0:
                        plan.append(n)

    return plan[-1]