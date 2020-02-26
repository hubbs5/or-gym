from pyomo.environ import *
from pyomo.opt import SolverFactory

def solve_math_program(model, solver='glpk', print_results=True):

	solver = SolverFactory(solver)
	results = solver.solve(model, tee=print_results)
	return model, results