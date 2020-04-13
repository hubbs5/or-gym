from pyomo.environ import *
from pyomo.opt import SolverFactory

def build_vm_packing_model(env):
	assert env.spec.id == 'VMPacking-v0', \
        '{} received. Heuristic designed for VMPacking-v0.'.format(env.spec.id)

	m = ConcreteModel()

	m.n = Set(initialize=[i for i in range(env.n_pms)]) # Num Physical Machines
	m.v = Set(initialize=[i for i in range(env.step_limit)]) # Num Virtual Machines
	m.t = Set(initialize=[i for i in range(env.step_limit)])
	m.cpu_demand = Param(m.t,
	    initialize={i: j[0] for i, j in enumerate(env.demand)})
	m.mem_demand = Param(m.t,
	    initialize={i: j[1] for i, j in enumerate(env.demand)})
	m.durations = Param(m.v,
	    initialize={i: env.step_limit - i for i in range(env.step_limit)})
	min_demand = min(m.mem_demand[t] for t in m.t)
	m.cpu_limit = env.cpu_capacity
	m.mem_limit = env.mem_capacity
	m.x = Var(m.n, m.v, m.t, within=Binary) # Assign VM's to machines
	m.y = Var(m.n, m.v, within=Binary)
	m.z = Var(m.n, m.t, within=Binary)

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
	def machine_assignment(m, v):
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