import pandas as pd
import cplex
import numpy as np
import matplotlib.pyplot as plt

# flatten a list of lists (depth = 2)
flatten = lambda l: [item for sublist in l for item in sublist]

# load data
xlsx = pd.ExcelFile('./data_if_18/dowjones.xlsx')

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

# r_a(i) = arithmetic mean of returns of asset i
#r_a = returns.mean(0)

# variance of returns
#r_var = returns.var(0)

### Q = covariance matrix
#Q = returns.cov()

### Correlation matrix using Pearson distance : ro_i_j = cov(i,j) / sigma_i * sigma_j
rho = returns.corr(method='pearson')

############################################################################################################################
##########################################  Implementing the ILP model  ####################################################
############################################################################################################################

# parameter @q : number of assets to include in the index fund
q = 10

# Objective function : MAX sum_i_j rho_i_j * x_i_j   | i=1..n ; j=1..n
# variables: xij and yj
x_i_j = [[0] * N for i in range(N)]
for i in range(N):
    for j in range(N):
        x_i_j[i][j] = "x_"+str(i)+"_"+str(j)

y_j = ["y_"+str(i) for i in range(N) ]

# variables coefficients in the objective function is the correlation : rho
# bounds: xij and yj are binary
ub_x = [[1] * N for i in range(N)]
lb_x = [[0] * N for i in range(N)]
ub_y = [1] * N
lb_y = [0] * N

###### constraints
# sum_j (y_j) = q
# sum_j (x_i_j) = 1 for each i=1..n
# x_i_j - y_j <= 0 for each i,j = 1..n

# right hand side = q , 1 (n times) , 0 (n*n times) 
rhs = [q] + N * [1] + [0] *N*N

# sense : = , = (n times) , <= (n*n times)
senses = "E" + "E" * N + "L" * N * N

# rows
# sum_j (y_j) = q                               # 1 line
# sum_j (x_i_j) = 1     for each i=1..n         # n lines
# x_i_j - y_j <= 0      for each i,j = 1..n     # n*n lines

_rows = [
    [ y_j , [1] * N ]
]

for i in range(N):
    _rows.append(
        [ x_i_j[i] , [1] * N ]
    )

for i in range(N):
    for j in range(N):
        _rows.append(
            [ [ x_i_j[i][j] , y_j[j]  ] , [1 , -1] ]
        )

_prob = cplex.Cplex()
_prob.objective.set_sense(_prob.objective.sense.maximize)
## add variables. 
# PS: flatten matrices before adding them
# PS: keep same order : flat(x_i_j) then y_j
_prob.variables.add(
    obj= rho.values.flatten().tolist() + [0]*N , # y_j are not in the objective function hence their coeff is 0
    lb= flatten(lb_x) + lb_y,
    ub= flatten(ub_x) + ub_y,
    types = [_prob.variables.type.integer] * (N*N + N),
    names= flatten(x_i_j) + y_j
)

_prob.linear_constraints.add(
    lin_expr= _rows , 
    senses= senses, 
    rhs= rhs
)

_prob.solve()

print "solution status"
print _prob.solution.status[ _prob.solution.get_status() ]

print "Objective Solution value"
print _prob.solution.get_objective_value()

print "Parameters values"
values = _prob.solution.get_values()
print values

y = values[-N:]

index = []
for i in range(N):
    if y[i] == 1.0:
        index.append(i)
        print "asset", assets[i] , "is in the portfolio"

###############################################################################################################################
###################################################### compute weights in the index fund ######################################
###############################################################################################################################

# TODO: CAC40 => cap-weighted
# TODO: Dow-Jones => price-weighted

## w_j = sum_over_i (V_i * x_i_j)
# Dow-Jones => price weighted => V_i =  price_i / sum_i prices_i

# x_i_j values of the solver
x_values = values[0: N*N]

# compute market weight at a given date: t<= T
t = T-1

last_prices = prices.iloc[t].values
sum_prices = sum(last_prices)
market_weights = []
for p in last_prices:
    market_weights.append(p / sum_prices)

# compute index weight
# list x_i_j to matrix
shape = (N, N)
x_i_j = np.array( values[0:N*N] )
x_i_j = x_i_j.reshape(shape)

# wj is the total market value of the stocks represented by stock j in the fund
wj = []
for j in range(N):
    _wj = 0
    for i in range(N):
        _wj+= market_weights[i] * x_i_j[i][j]
    wj.append(_wj)

## adjust the index-fund weights so that the total equals 1
sum_fund = sum(wj)
for i in range(N):
    wj[i] = wj[i] / sum_fund

###############################################################################################################################
############################################ portfolio and market performance comparison ######################################
###############################################################################################################################

# read dow jones market performance csv
_csv = pd.read_csv('./data_if_18/dowjones_average.csv')
# take the close column which the closing price of the market
market_prices = _csv['Close']
market_returns = []
for i in range(1,t):
    market_returns.append( market_prices[i]/market_prices[i-1] -1  )

returns = returns.values
index_fund_returns = []
for i in range(0, t-1):
    index_fund_return = 0
    for x in range(N):
        index_fund_return += wj[x] * returns[i][x]
    index_fund_returns.append(index_fund_return)

###################################################
# compare index_fund_returns and market_returns
for i in range(100,2101 , 100):
    plt.plot(index_fund_returns[0:i] , 'r' , label='index fund performance')
    plt.plot(market_returns[0:i] , 'b' , label='market performance')
    plt.title('index fund vs market performances')
    plt.legend()
    png_title = './figures/comparison_'+str(i)+'_dates'
    plt.savefig(png_title)
    plt.close()

