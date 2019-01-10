import pandas as pd
import cplex

# load data
xlsx = pd.ExcelFile('./data_if_18/CAC40.xlsx')

# parse the required sheet
data = xlsx.parse(0)
dates = data['Date']
# keep only prices
prices = data.drop( columns=['Date'] )
# temporal horizon = number of dates
T = len(prices)
# list of assets
assets = prices.columns
# N = number of assets
N = len(assets)
# r(i,t) = return of asset i at date t
returns = pd.DataFrame(data = [] )
for s in assets:
    for t in range(1,T):
        returns.at[t,s] = (prices.at[t,s] - prices.at[t-1,s]) / prices.at[t-1,s]

# mu = arithmetic mean of returns of each asset
mu = returns.mean(0)
# variance of returns of each asset
#vars = returns.var(0)
### Q = covariance matrix
Q = returns.cov().values


####################################################################################################################################
############################################ implementing Markowitz model ##########################################################
####################################################################################################################################
# @input : minimum target return 
R = 0.05

# variable names : x_0 -> x_N
names = ["x_"+str(i) for i in range(N) ]
# variables indices
x_i = [i for i in range(N)] 

# quadratic objective
qmat = []
for i in range(N):
    qmat.append( [ x_i , Q[i] ] )

_ub = [cplex.infinity] * N
_lb = [0.0] * N

# constraints
