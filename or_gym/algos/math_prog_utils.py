from pyomo.environ import *
from pyomo.opt import SolverFactory

def solve_math_program(model, solver='glpk', print_results=True):

    solver = SolverFactory(solver)
    results = solver.solve(model, tee=print_results)
    return model, results

def extract_vm_packing_plan(model):
    plan = []
    for v in m.v:
        for t in m.t:
            if v == t:
                for n in m.n:
                    if m.x[n, v, t].value is None:
                        continue
                    if m.x[n, v, t].value > 0:
                        plan.append([n, v, t])

    return np.vstack(plan)