from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from scipy.optimize import minimize
import itertools

def solve_math_program(model, solver='glpk', print_results=True, 
                        warmstart = False, mapping_env = None, mapping_z = None):
    '''
    Solves mathematical program using pyomo
    
    model = [Pyomo model]
    solver = [String] optimization solver to use
    print_results = [Boolean] should the results of the optimization be printed?
    '''
    
    solver = SolverFactory(solver)
    if warmstart:
        model = warm_start_nv(model, mapping_env, mapping_z)
        results = solver.solve(model, tee=print_results, warmstart=warmstart)
    else:
        results = solver.solve(model, tee=print_results)
        return model, results

def warm_start_nv(model, mapping_env, mapping_z):
    '''
    For NewsVendor
    
    Uses finalized simulation with the base stock policy to provide a warm start to
    the base stock MIP.
    
    model = [Pyomo model]
    mapping_env = [NewsVendor Environment] that has been run until completion
        NOTE: the mapping environment must have the same demand profile as the MIP model
    mapping_z = [list] base_stock used in mapping_env
    '''
    
    #extract solution from mapping_env
    I = mapping_env.I
    T = mapping_env.T
    R = mapping_env.R 
    S = mapping_env.S
    B = mapping_env.B 
    LS = mapping_env.LS
    P = mapping_env.P
    N = mapping_env.num_periods
    M = mapping_env.num_stages
    backlog = mapping_env.backlog
    z = np.array(list(mapping_z))
    x = np.diff(z, prepend = [0])
    c = mapping_env.supply_capacity
    D = mapping_env.D
    
    #populate model
    for n in range(N+1): #mip.n1
        for m in range(M): #mip.m
            if m < M-1: #mip.m0
                model.I[n,m] = I[n,m]
                model.T[n,m] = T[n,m]
                if n < N: #mip.n
                    model.R[n,m] = R[n,m]
                    if n>0:
                        R1 = max(0, z[m] - np.sum(I[n,:m+1] + T[n,:m+1] - B[n-1,:m+1]) + B[n-1,m+1])
                        model.R1[n,m] = R1
                    else:
                        R1 = max(0, z[m] - np.sum(I[n,:m+1] + T[n,:m+1]))
                        model.R1[n,m] = R1
                    if R1 == 0:
                        model.y[n,m] = 0
                    else:
                        model.y[n,m] = 1
                    if R[n,m] == R1:
                        model.y1[n,m] = 1
                        model.y2[n,m] = 0
                        if M > 2:
                            model.y3[n,m] = 0
                    elif R[n,m] == c[m]:
                        model.y1[n,m] = 0
                        model.y2[n,m] = 1
                        if M > 2:
                            model.y3[n,m] = 0
                    else:
                        model.y1[n,m] = 0
                        model.y2[n,m] = 0
                        if M > 2:
                            model.y3[n,m] = 1
            if n < N: #mip.n
                model.S[n,m] = S[n,m]
                if backlog:
                    model.B[n,m] = B[n,m]
                else:
                    model.LS[n,m] = LS[n,m]
                if n>0:
                    if S[n,m] == D[n] + B[n-1,m]:
                        model.y4[n,m] = 0
                    else:
                        model.y4[n,m] = 1
                else:
                    if S[n,m] == D[n]:
                        model.y4[n,m] = 0
                    else:
                        model.y4[n,m] = 1
        if n < N: #mip.n
            model.P[n] = P[n]
    
    for m in range(M-1):
        model.z[m] = int(z[m])
        model.x[m] = int(x[m])
    
    return model
    
def solve_dfo_program(env, fun, online = False, local_search = False, print_results=True):
    '''
    Optimize base stock level on a simulated sample path. Minimization is done by Powell's method.
    Powell's method is used since the objective function is not smooth.
    
    env_args = [list] arguments for simulation environment on which to perform SPA.
    fun = [function] function over which to optimize (i.e. nv_min_model or onv_min_model)
    online = [Boolean] should the optimization be run in online mode?
    local_search = [Boolean] search neighborhood around NLP relaxation to get integer solution (by enumeration).
    print_results = [Boolean] should the results of the optimization be printed?
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
    
    #print results
    if print_results:
        print(res)
    
    return res