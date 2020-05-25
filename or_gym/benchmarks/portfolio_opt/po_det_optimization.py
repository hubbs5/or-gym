import numpy as np 
from pyomo.environ import *
from pyomo.opt import SolverFactory 

po = ConcreteModel()

investment_horizon = 10

assets = [a for a in range(3)]
periods = [p for p in range(investment_horizon+1)]

#Parameters 
initial_cash = 150
initial_assets = [0, 0, 0]
buy_cost = [0.045, 0.025, 0.035]
sell_cost = [0.040, 0.020, 0.030]
asset_prices = np.random.rand(len(assets), investment_horizon+1)+0.5

#Variables 
po.Cash_Quantity = Var(periods, domain=NonNegativeReals)
po.Cash_Quantity[0].fix(initial_cash)
po.Asset_Quantities = Var(assets, periods, domain=NonNegativeReals)
for a in assets: 
	po.Asset_Quantities[a,0].fix(initial_assets[0])
po.Asset_Sell = Var(assets, periods[1:], domain=NonNegativeReals)
po.Asset_Buy = Var(assets, periods[1:], domain=NonNegativeReals)

#Constraints 
def CashAccounting(po, p): 
	return po.Cash_Quantity[p] == po.Cash_Quantity[p-1] \
	+ sum((1-sell_cost[a])*asset_prices[a][p]*po.Asset_Sell[a,p] - \
		(1+buy_cost[a])*asset_prices[a][p]*po.Asset_Buy[a,p] for a in assets)

def AssetBalance(po, a, p): 
	return po.Asset_Quantities[a,p] == po.Asset_Quantities[a,p-1] \
	- po.Asset_Sell[a,p] + po.Asset_Buy[a,p]

po.CashAccounting = Constraint(periods[1:], rule=CashAccounting)
po.AssetBalance = Constraint(assets, periods[1:], rule=AssetBalance)

#Objective
def PortfolioValue(po): 
	return po.Cash_Quantity[investment_horizon] + \
	sum(asset_prices[a][-1]*po.Asset_Quantities[a,investment_horizon] for a in assets)

po.PortfolioValue = Objective(rule=PortfolioValue, sense=maximize)

opt = SolverFactory('gurobi')
results = opt.solve(po, tee=False, keepfiles=False)

# print(results)
# po.Cash_Quantity.pprint()
# po.Asset_Quantities.pprint()

