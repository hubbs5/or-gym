import numpy as np 
from pyomo.environ import *
from pyomo.opt import SolverFactory 

po = ConcreteModel()

investment_horizon = 10

#Define such that element at each index matches indexing of assets and periods in optimization formulation 
assets = [a+1 for a in range(3)]
periods = [p for p in range(investment_horizon+1)]

#Parameters 
initial_cash = 100
initial_assets = [0, 0, 0]
buy_cost = [0.045, 0.025, 0.035]
sell_cost = [0.040, 0.020, 0.030]
asset1mean = np.array([1.25, 2, 4, 5, 3, 2, 3, 6, 9, 7]).reshape(1, -1) #up and down all the way 
asset2mean = np.array([5, 3, 2, 2, 1.25, 4, 5, 6, 7, 8]).reshape(1, -1) #down intially then up 
asset3mean = np.array([3, 5, 6, 9, 10, 8, 4, 2, 1.25, 4]).reshape(1, -1) #up initially then down 
asset_prices = np.vstack([asset1mean, asset2mean, asset3mean])

#Variables 
po.Cash = Var(periods, domain=NonNegativeReals)
po.Cash[0].fix(initial_cash)
po.Holdings = Var(assets, periods, domain=NonNegativeReals)
for a in assets: 
	po.Holdings[a,0].fix(initial_assets[0])

po.Asset_Sell = Var(assets, periods[1:], domain=NonNegativeReals)
po.Asset_Buy = Var(assets, periods[1:], domain=NonNegativeReals)

#Constraints 
def CashAccounting(po, p): 
	return po.Cash[p] == po.Cash[p-1] \
	+ sum((1-sell_cost[a-1]) * asset_prices[a-1][p-1] * po.Asset_Sell[a,p] - \
		(1+buy_cost[a-1]) * asset_prices[a-1][p-1]* po.Asset_Buy[a,p] for a in assets)

def AssetBalance(po, a, p): 
	return po.Holdings[a,p] == po.Holdings[a,p-1] \
	- po.Asset_Sell[a,p] + po.Asset_Buy[a,p]

po.CashAccounting = Constraint(periods[1:], rule=CashAccounting)
po.AssetBalance = Constraint(assets, periods[1:], rule=AssetBalance)

#Objective
def PortfolioValue(po): 
	return po.Cash[investment_horizon] + \
	sum(asset_prices[a-1][-1] * po.Holdings[a, investment_horizon] for a in assets)

po.PortfolioValue = Objective(rule=PortfolioValue, sense=maximize)

opt = SolverFactory('gurobi')
results = opt.solve(po, tee=False, keepfiles=False)

po.Cash.pprint()
po.Holdings.pprint()
po.Asset_Sell.pprint()
po.Asset_Buy.pprint()

