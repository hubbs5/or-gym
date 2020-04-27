from pyomo.environ import *

def build_online_vm_opt(env):
    m = ConcreteModel()
    
    t0 = env.current_step
    m.n = Set(initialize=[i for i in range(env.n_pms)]) # Num Physical Machines
    # One new VM at each time step
    m.v = Set(initialize=[v for v in range(t0+1)])
    m.t = Set(initialize=[t for t in range(env.step_limit)])
    
    m.cpu_demand = Param(m.v,
        initialize={i: j[1] for i, j in enumerate(env.demand[:t0+1])})
    m.mem_demand = Param(m.v,
        initialize={i: j[2] for i, j in enumerate(env.demand[:t0+1])})
    m.durations = Param(m.v,
        initialize={v: env.step_limit - v for v in m.v})
    m.cpu_limit = env.cpu_capacity
    m.mem_limit = env.mem_capacity

    m.x = Var(m.n, m.v, m.t, within=Binary) # Assign VM's to machines
    m.y = Var(m.n, m.v, within=Binary)
    m.z = Var(m.n, m.t, within=Binary) # Number of Machine
    
    # Fix variables for pre-existing values
    if t0 > 0:
        for k in env.assignment:
            n = env.assignment[k]
            for t in m.t:
                if t >= k:
                    m.x[n, k, t].fix(1)
                else:
                    m.x[n, k, t].fix(0)
            m.y[n, k].fix(1)
        
    @m.Constraint(m.n, m.t)
    def cpu_constraint(m, n, t):
        return sum(m.x[n, v, t] * m.cpu_demand[v] 
                   for v in m.v) - m.cpu_limit <= 0

    @m.Constraint(m.n, m.t)
    def mem_constraint(m, n, t):
        return sum(m.x[n, v, t] * m.mem_demand[v]
                   for v in m.v) - m.mem_limit <= 0

    @m.Constraint(m.n, m.v)
    def duration_constraint(m, n, v):
        return sum(m.x[n, v, t] 
                for t in m.t)- m.y[n, v] * m.durations[v] == 0

    @m.Constraint(m.v)
    def vm_machine_assignment(m, v):
        return sum(m.y[n, v] for n in m.n) == 1

    @m.Constraint(m.v, m.t)
    def assignment_constraint(m, v, t):
        if t >= v:
            return sum(m.x[n, v, t] for n in m.n) <= 1
        else:
            return sum(m.x[n, v, t] for n in m.n) == 0

    @m.Constraint(m.n, m.v, m.t)
    def time_constraint(m, n, v, t):
        return (m.z[n, t] - m.x[n, v, t]) >= 0

    # Maximize PM Packing
    m.obj = Objective(expr=(
        sum(sum(m.x[n, v, t] * (m.cpu_demand[v] + m.mem_demand[v])
            for v in m.v) - 2 * m.z[n, t]
            for n in m.n
            for t in m.t)),
        sense=maximize)
    return m