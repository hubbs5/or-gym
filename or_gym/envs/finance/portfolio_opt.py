import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from copy import copy

class PortfolioOptEnv(gym.Env): 
    '''
    Portfolio Optimization Problem 

    Instance: Multi-Period Asset Allocation Problem, Dantzing & Infager, 1993 

    The Portfolio Optimization (PO) Problem is a problem that seeks to optimize 
    the distribution of assets in a financial portfolio to with respect to a desired 
    financial metric (e.g. maximal return, minimum risk, etc.). 

    In this particular instance by Dantzing & Infager, the optimizer begins with a 
    quantity of cash and has the opportunity to purchase or sell 3 other assets in each  
    of 10 different investement periods. Each transaction incurs a cost and prices of 
    the 3 assets are subject to change over time. Cash value is consant (price = 1). 
    The objective is to maximize the amount of wealth (i.e. the sum total of asset values)
    at the end of the total investment horizon.

    The episodes proceed by the optimizer deciding whether to buy or sell each asset
    in each time period. The episode ends when either all 10 periods have passed or 
    if the amount of any given asset held becomes negative.  

    Observation:
        Type: Box(7)
        "asset prices" (idx 0,1,2): array of asset prices [cash, asset1, asset2, asset3]
        "asset quantities" (idx 3,4,5): array of asset quantities [cash, asset1, asset2, asset3]
        "total wealth" (idx 6): current total wealth (sum of price*quantity for each asset)
    

    Actions:
        Type: Box (3)
        "asset 1 transaction amount" (idx 0): Buy/sell up to n amount of asset 1; 0: no transaction
        "asset 2 transaction amount" (idx 1): Buy/sell up to n amount of asset 2; 1-n: Sell j "in" [1,n] amt of asset
        "asset 3 transaction amount" (idx 2): Buy/sell up to n amount of asset 3; n+1-2n: Buy j "in" [1,n] amt of asset

    Reward:
        Change in total wealth from previous period or [-max(asset price of all assets) *  maximum transaction size]
        if an asset quantity becomes negative, at which 
        point the episode ends.

    Starting State:
        Starting amount of cash and wealth and prices. 

    Episode Termination:
        Negative asset quantity or traversal of investment horizon. 
    '''

    def __init__(self, *args, **kwargs): 
        
        #Immutable Parameters 
        self.num_assets = 3 
        self.max_transaction_size = 25
        self.cash_price = 1 
        self.initial_cash = 150
        self.initial_assets = np.zeros(3)
        self.buy_cost = np.array([0.045, 0.025, 0.035])
        self.sell_cost = np.array([0.040, 0.020, 0.030])
        self._max_reward = 400
        self.investment_horizon = 10 
        self.asset_prices_means = np.random.rand(
            self.investment_horizon + 1, self.num_assets) + 0.5
        self.asset_price_variance = np.ones(self.num_assets) * 0.25
        self.max_steps = copy(self.investment_horizon)

        #Define observation and action spaces
        self.observation_space = spaces.Box(7) 
        self.action_space = spaces.Box(3)
        #set seed 
        self.seed()
        #reset state 
        self.state = self.reset()

    def reset(self): 
        self.asset_quantities = np.array([self.initial_cash, self.initial_assets[0], self.initial_assets[1], \
            self.initial_assets[2]])
        self.total_wealth = self.initial_cash*self.cash_price

        self.step_counter = 0 
        self._update_state()

        return self.state

    def _get_obs(self):
        return self.state

    def _update_state(self): 
        self.asset_prices = np.concatenate((np.array([self.cash_price]), \
         np.random.normal(self.asset_prices_means,self.asset_prices_variance)))
        self.total_wealth = self.current_total_wealth
        self.state = np.array([
            self.asset_prices, 
            self.asset_quantities, 
            self.total_wealth])

    def step(self, action): 

        
        #Update asset and cash quantities 
        for idx, a in enumerate(action): 
            if a in range(self.max_transaction_size): 
                self.asset_quantities[idx+1] -= a 
                self.asset_quantities[0] += (1-self.sell_cost[idx])*self.asset_prices[idx]*a 

            else: 
                self.asset_quantities[idx+1] += a-self.max_transaction_size
                self.asset_quantities[0] -= (1+self.buy_cost)*self.asset_prices[idx]*(a-self.max_transaction_size)

        #Calculate reward 
        current_total_wealth = 0 
        for idx, a in enumerate(self.asset_prices): 
            self.current_total_wealth += self.asset_prices[idx]*self.asset_quantities[idx]
        if np.all(self.asset_quantities >= 0): 
            reward = self.current_total_wealth - self.total_wealth
            done = False
            Termination = "Termination Condition: No Termination"
        else: 
            reward = -max(self.asset_prices)*self.max_transaction_size
            done = True 
            Termination = "Termination Condition: Negative Asset Value"

        self.step_counter += 1

        if self.step_counter > self.max_steps: 
            done = True
            Termination = "Termination Condition: End of Horizon"

        self._update_state()

        return self.state, reward, done, {"Status": Termination}

    def sample_action(self): 
        return np.array(np.random.choice(range(2*self.max_transaction_size+1), size=len(self.asset_prices-1)))