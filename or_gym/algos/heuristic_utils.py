import gym
import or_gym
import numpy as np
from scipy.optimize import minimize
import itertools

def solve_min_program(env, fun, local_search = False):
    '''
    Optimize base stock level on a simulated sample path. Minimization is done by bisection.
    Bisection is used since objective function is not smooth.
    
    env_args = [list] arguments for simulation environment on which to perform SPA.
    fun = [function] function over which to optimize (i.e. nv_min_model or onv_min_model)
    use_CR = [Boolean] use critical ratio optimal base-stock levels for the initial value.
    local_search = [Boolean] search neighborhood around NLP relaxation to get integer solution (by enumeration).
    '''      
    
    x = np.ones(env.num_stages - 1) #initial inventory levels

    #run optimization
    res = minimize(fun = fun, x0 = x, args = env, method = 'Powell')
    xopt = res.x[()]
    fopt = -res.fun
    
    #local search to get integer solution
    if local_search:
        xopt_f = np.floor(xopt)
        xopt_c = np.ceil(xopt)
        X = np.column_stack((xopt_f,xopt_c))
        xlist = list(itertools.product(*X)) #2^|x| combinations around NLP relaxation optimum
        fopt = -np.Inf #initialize incumbent
        for x in xlist:
            f = -fun(x,env)
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