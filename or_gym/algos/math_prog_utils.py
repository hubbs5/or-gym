from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from scipy.optimize import minimize
import itertools

def solve_math_program(model, solver='glpk', print_results=True):
    '''
    Solves mathematical program using pyomo
    
    model = [Pyomo model]
    solver = [String] optimization solver to use
    print_results = [Boolean] should the results of the optimization be printed?
    '''
    
    solver = SolverFactory(solver)
    results = solver.solve(model, tee=print_results)
    return model, results
    
def solve_min_program(env, fun, online = False, local_search = False):
    '''
    Optimize base stock level on a simulated sample path. Minimization is done by bisection (Powell's method).
    Bisection is used since objective function is not smooth.
    
    env_args = [list] arguments for simulation environment on which to perform SPA.
    fun = [function] function over which to optimize (i.e. nv_min_model or onv_min_model)
    online = [Boolean] should the optimization be run in online mode?
    local_search = [Boolean] search neighborhood around NLP relaxation to get integer solution (by enumeration).
    '''      
    
    x = np.ones(env.num_stages - 1) #initial inventory levels

    #run optimization
    res = minimize(fun = fun, x0 = x, args = (env,online), method = 'Powell')
    xopt = res.x[()]
    fopt = -res.fun
    
    #local search to get integer solution (if solution is not already integer)
    if local_search & (np.sum(np.mod(xopt,1)) != 0):
        xopt = np.max(np.column_stack((np.zeros(env.num_stages - 1),xopt)),axis=1) #correct negative values to 0
        xopt_f = np.floor(xopt)
        xopt_c = np.ceil(xopt)
        X = np.column_stack((xopt_f,xopt_c))
        xlist = list(itertools.product(*X)) #2^|x| combinations around NLP relaxation optimum
        fopt = -np.Inf #initialize incumbent
        for x in xlist:
            f = -fun(x,env,online)
            if f>fopt: #update incumbent
                fopt = f
                xopt = x
    
    #calculate base stock level
    zopt = np.cumsum(xopt) 
    zopt = np.round(zopt)
    xopt = np.round(xopt)
    
    #store results
    res.xopt = xopt
    res.zopt = zopt
    res.fopt = fopt
    
    return res