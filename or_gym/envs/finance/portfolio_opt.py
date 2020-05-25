import numpy as np
import gym
from gym import spaces, logger
from gym.utils import seeding
from or_gym.utils.env_config import *
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
        Type: Box(9)
        "asset prices" (idx 0, 1, 2, 3): array of asset prices [cash, asset1, asset2, asset3]
        "asset quantities" (idx 4, 5, 6, 7): array of asset quantities [cash, asset1, asset2, asset3]
        "total wealth" (idx 8): current total wealth (sum of price*quantity for each asset)
    

    Actions:
        Type: Box (3)
        "asset 1 transaction amount" (idx 0): Buy/sell up to n amount of asset 1; 
        "asset 2 transaction amount" (idx 1): Buy/sell up to n amount of asset 2; 
        "asset 3 transaction amount" (idx 2): Buy/sell up to n amount of asset 3; 

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
        self.num_assets = 3
        self.initial_cash = 100
        self.cash = copy(self.initial_cash)
        self.buy_cost = np.array([0.045, 0.025, 0.035])
        self.sell_cost = np.array([0.04, 0.02, 0.03])
        self.step_limit = 10
        assign_env_config(self, kwargs)
        self.asset_price_means = asset_price_means.T
        # self.asset_price_means = (np.random.randint(10, 50, self.num_assets) \
        #     * np.ones((self.step_limit, self.num_assets))).T
        self.asset_price_var = np.ones(self.asset_price_means.shape)
        
        # Cash on hand, asset prices, num of shares, portfolio value
        self.obs_length = 1 + 2 * self.num_assets
#         self.observation_space = spaces.Dict({
#             "action_mask": spaces.Box(0, 1, shape=(self.num_assets,)),
#             "avail_actions": spaces.Box(0, 1, shape=(self.num_assets,)),
#             "state": spaces.Box(-1000, 1000, shape=(self.obs_length,))
#         })
        self.observation_space = spaces.Box(0, 10000, shape=(self.obs_length,))
        self.action_space = spaces.Box(-1000, 1000, shape=(self.num_assets,))
        
        self.reset()
        
    def reset(self):
        self.step_count = 0
        self.asset_prices = self._generate_asset_prices()        
        self.holdings = np.zeros(self.num_assets)
        self.cash = copy(self.initial_cash)
        self.state = np.hstack([
            self.initial_cash,
            self.asset_prices[:, self.step_count],
            self.holdings])
        return self.state
    
    def _generate_asset_prices(self):
        asset_prices = np.array([np.random.normal(mu, sig) for mu, sig in 
            zip(self.asset_price_means.flatten(), self.asset_price_var.flatten())]
            ).reshape(self.asset_price_means.shape)
        # Zero out negative asset prices and all following prices - implies
        # equity is bankrupt and worthless.
        zero_vals = np.vstack(np.where(asset_prices<0))
        cols = np.unique(zero_vals[0])
        for c in cols:
            first_zero = zero_vals[1][np.where(zero_vals[0]==c)[0].min()]
            asset_prices[c,first_zero:] = 0
        return asset_prices
    
    def step(self, action):
        # Round actions to integer values
        action = np.round(action)
        asset_prices = self.asset_prices[:, self.step_count].copy()
        for idx, a in enumerate(action):
            if a == 0:
                continue
            # Sell a shares of asset
            elif a < 0:
                a = np.abs(a)
                if a > self.holdings[idx]:
                    a = self.holdings[idx]
                self.holdings[idx] -= a
                self.cash += asset_prices[idx] * a * (1 - self.sell_cost[idx])
            # Buy a shares of asset
            elif a > 0:
                purchase_cost = asset_prices[idx] * a * (1 + self.buy_cost[idx])
                if self.cash < purchase_cost:
                    a = np.floor(self.cash / (
                        asset_prices[idx] * (1 + self.buy_cost[idx])))
                    purchase_cost = asset_prices[idx] * a * (1 + self.buy_cost[idx])
                self.holdings[idx] += a
                self.cash -= purchase_cost
                
        # Return total portfolio value as reward
        reward = np.dot(asset_prices, self.holdings) + self.cash
        self.step_count += 1
        if self.step_count >= self.step_limit:
            done = True
        else:
            self._update_state()
            done = False
            
        return self.state, reward, done, {}
    
    def _update_state(self):
        self.state = np.hstack([
            self.cash,
            self.asset_prices[:, self.step_count],
            self.holdings
        ])

    # def __init__(self, *args, **kwargs): 
        
    #     #Immutable Parameters 
    #     self.num_assets = 3 
    #     self.max_transaction_size = 25
    #     self.cash_price = 1 
    #     self.initial_cash = 150
    #     self.initial_assets = np.zeros(3)
    #     self.buy_cost = np.array([0.045, 0.025, 0.035])
    #     self.sell_cost = np.array([0.040, 0.020, 0.030])
    #     self._max_reward = 400
    #     self.investment_horizon = 10 
    #     self.asset_prices_means = asset_prices_means
    #     self.asset_price_variance = np.ones(self.num_assets) * 0.25
    #     self.max_steps = copy(self.investment_horizon)

    #     #Define observation and action spaces
    #     self.observation_space = spaces.Box(-10000, 10000, shape=(9,)) 
    #     self.action_space = spaces.Box(-self.max_transaction_size, self.max_transaction_size, 
    #         shape=(self.num_assets,))
    #     #set seed 
    #     self.seed()
    #     #reset state 
    #     self.state = self.reset()

    # def reset(self): 
    # 	self.asset_quantities = np.array([self.initial_cash, self.initial_assets[0], self.initial_assets[1], \
    #         self.initial_assets[2]])
    # 	self.total_wealth = self.initial_cash*self.cash_price
    # 	self.step_counter = 0 
    # 	self._update_state()

    # 	return self.state

    # def _get_obs(self):
    #     return self.state

    # def _update_state(self): 
    #     if self.step_counter > 0:
    #         self.total_wealth = self.current_total_wealth
    #     self.asset_prices = np.concatenate((np.array([self.cash_price]), \
    #         np.random.normal(self.asset_prices_means[self.step_counter], self.asset_price_variance)))
    #     self.state = np.hstack([
    #         self.asset_prices, 
    #         self.asset_quantities, 
    #         self.total_wealth])

    # def step(self, action):        
    #     #Update asset and cash quantities 
    #     for idx, a in enumerate(action): 
    #         if a < 0: 
    #             self.asset_quantities[idx+1] += a 
    #             self.asset_quantities[0] -= (1-self.sell_cost[idx])*self.asset_prices[idx]*a 

    #         else: 
    #             self.asset_quantities[idx+1] += a
    #             self.asset_quantities[0] -= (1+self.buy_cost[idx])*self.asset_prices[idx]*a

    #     #Calculate reward 
    #     self.current_total_wealth = 0 
    #     for idx, a in enumerate(self.asset_prices): 
    #         self.current_total_wealth += self.asset_prices[idx]*self.asset_quantities[idx]
    #     if np.all(self.asset_quantities >= 0): 
    #         reward = self.current_total_wealth - self.total_wealth
    #         done = False
    #         Termination = "Termination Condition: No Termination"
    #     else: 
    #         reward = -max(self.asset_prices)*self.max_transaction_size
    #         done = True 
    #         Termination = "Termination Condition: Negative Asset Value"

    #     self.step_counter += 1

    #     if self.step_counter > self.max_steps: 
    #         done = True
    #         Termination = "Termination Condition: End of Horizon"
    #     else: 
    #         self._update_state()

    #     return self.state, reward, done, {"Status": Termination}

    # def sample_action(self): 
    #     return np.random.uniform(low=-self.max_transaction_size, 
    #         high=self.max_transaction_size,size=self.num_assets)


#num_assets = 3
#investment_horizon = 10 
#asset_price_means = np.random.rand(investment_horizon + 1, num_assets) + 0.5
asset_price_means = np.array([
    [0.729104  , 0.70066482, 1.33728305],
    [0.71028955, 1.15127388, 0.65365377],
    [0.83731888, 0.78674174, 1.14186928],
    [0.83644462, 0.97910886, 0.94767697],
    [0.69826764, 1.14386794, 0.94392694],
    [0.69017948, 0.86546669, 0.82813273],
    [0.61135848, 0.72119583, 0.70126934],
    [0.58991467, 0.86416669, 1.18881049],
    [1.48227405, 1.41814408, 0.96752138],
    [0.5027847 , 0.5380547 , 0.62442277],
    [0.56073499, 1.27841103, 1.18236989]])