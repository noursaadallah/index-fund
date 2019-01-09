import pandas as pd
import cplex

# flatten a list of lists (depth = 2)
flatten = lambda l: [item for sublist in l for item in sublist]

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
# r(i,t) = interest of asset i at date t
interests = pd.DataFrame(data = [] )
for s in assets:
    for t in range(1,T):
        interests.at[t,s] = (prices.at[t,s] - prices.at[t-1,s]) / prices.at[t-1,s]

# r_a(i) = arithmetic mean of interests of asset i
#r_a = interests.mean(0)

# variance of interests
#r_var = interests.var(0)

### Q = covariance matrix
Q = interests.cov()

### Correlation matrix using Pearson distance : ro_i_j = cov(i,j) / sigma_i * sigma_j
rho = interests.corr(method='pearson')

############################################################################################################################
##########################################  Implementing the ILP model  ########################################################
############################################################################################################################

# parameter @q : number of assets to take
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
    obj= rho.values.flatten().tolist() + [0]*N , # y_j are not in the objective function hence 0
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
print _prob.solution.get_values()