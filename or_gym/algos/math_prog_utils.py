import pyomo.environ as pe
from pyomo.opt import SolverFactory
import time
import numpy as np
from scipy.optimize import minimize
import itertools
import or_gym

def solve_math_program(model, solver='glpk', solver_kwargs={}, print_results=False,
                        warmstart=False, warmstart_kwargs={}):
    '''
    Solves mathematical program using pyomo
    
    model = [Pyomo model]
    solver = [String] optimization solver to use
    solver_kwargs: [Dict] solver specific options (i.e. TimeLimit, MIPGap)
    print_results = [Boolean] should the results of the optimization be printed?
    warmstart = [Boolean] should math program be warm started?
    warmstart_kwargs:
        'mapping_env' = [InvManagement Environment] that has been run until completion
            NOTE: the mapping environment must have the same demand profile as the MIP model
        'mapping_z' = [list] base_stock used in mapping_env
        'online' = [Boolean] is the optimization being done online?
    '''
    
    solver = SolverFactory(solver) #create solver
    if len(solver_kwargs) > 0: # set solver keyword args
        solver.options = solver_kwargs
    if warmstart: #run warmstart
        model = warm_start_im(model, **warmstart_kwargs)
        results = solver.solve(model, tee=print_results, warmstart=warmstart)
        return model, results
    else: #run regular solve
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

def warm_start_im(model, mapping_env, mapping_z, online=False, perfect_information=False):
    '''
    For InvManagement
    
    Uses finalized simulation with the base stock policy to provide a warm start to
    the base stock MIP.
    
    model = [Pyomo model]
    mapping_env = [InvManagement Environment]
        NOTE: the mapping environment must have the same demand profile as the MIP model
    mapping_z = [list] base_stock used in mapping_env
    online = [Boolean] is the optimization being done online?
    perfect_information = [Boolean] is the optimization being run a perfect information model (no policy)?
    '''
    
    #extract base stock levels (z) and determine stage inventory levels (x)
    z = np.array(list(mapping_z))
    x = np.diff(z, prepend = [0])
    
    #extract arguments for env (copy of mapping_env)
    env_kwargs = {'periods': mapping_env.periods,
                  'I0': x,
                  'p': mapping_env.p,
                  'r': mapping_env.r,
                  'k': mapping_env.k,
                  'h': mapping_env.h,
                  'c': mapping_env.c,
                  'L': mapping_env.L,
                  'backlog': mapping_env.backlog,
                  'dist': mapping_env.dist,
                  'dist_param': mapping_env.dist_param,
                  'alpha': mapping_env.alpha,
                  'seed_int': mapping_env.seed_int}
    
    if online:
        #copy will only run until the last period
        env_kwargs['periods'] = mapping_env.period
    if perfect_information:
        #use initial inventory
        env_kwargs['I0'] = mapping_env.init_inv
        
    #create env
    if mapping_env.backlog:
        env = or_gym.make("InvManagement-v0",env_config=env_kwargs)
    else:
        env = or_gym.make("InvManagement-v1",env_config=env_kwargs)
        
    #run simulation    
    for t in range(env.num_periods):
        #take a step in the simulation using base stock policy
        env.step(action=env.base_stock_action(z=z)) 
        
    #extract solution to create warm start
    N = env.num_periods
    M = env.num_stages
    backlog = env.backlog
    c = env.supply_capacity
    
    I = env.I
    T = env.T
    R = env.R 
    S = env.S
    B = env.B 
    LS = env.LS
    P = env.P
    
    D = env.D
        
    #populate model
    for n in range(N+1): #mip.n1
        for m in range(M): #mip.m
            if m < M-1: #mip.m0
                model.I[n,m] = I[n,m]
                model.T[n,m] = T[n,m]
                if n < N: #mip.n
                    model.R[n,m] = R[n,m]
                    if not perfect_information:
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
                if not perfect_information:
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
            
    if not perfect_information:
        for m in range(M-1):
            model.z[m] = int(z[m])
            model.x[m] = int(x[m])
    
    return model
    
def solve_dfo_program(env, fun, online = False, local_search = False, print_results=True):
    '''
    Optimize base stock level on a simulated sample path. Minimization is done by Powell's method.
    Powell's method is used since the objective function is not smooth.
    
    env_args = [list] arguments for simulation environment on which to perform SPA.
    fun = [function] function over which to optimize (i.e. im_min_model or oim_min_model)
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