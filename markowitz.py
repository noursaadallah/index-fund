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
R = 0.0005

# variable names : x_0 -> x_N
_names = ["x_"+str(i) for i in range(N) ]
# variables indices
x_i = [i for i in range(N)] 

# quadratic objective
qmat = []
for i in range(N):
    qmat.append( [ x_i , Q[i] ] )

_ub = [cplex.infinity] * N
_lb = [0.0] * N

# constraints
# e_t * x = 1       s.t.  e = [1] * N
# mu_t * x >= R
_rhs = [1 , R]
_sense = "EG"
_rows = [
    [ _names , N*[1] ],
    [ _names , mu.values.tolist() ]
]

# init solver
_prob = cplex.Cplex()
# define objective sense : maximise or minimise
_prob.objective.set_sense(_prob.objective.sense.minimize)
# define objective variables (names + objective coeffs) and their bounds (upper and lower)
_prob.variables.add( obj=[0]*N , lb=_lb , ub=_ub , names= _names   )
# objective linear coeffs are zeroes
# set the quadratic part of the objective function : x_T * Q * x
_prob.objective.set_quadratic(qmat)

# add constraints
_prob.linear_constraints.add(lin_expr= _rows , senses=_sense , rhs= _rhs)

_prob.solve()

numrows = _prob.linear_constraints.get_num()
numcols = _prob.variables.get_num()

print "Solution status"
print _prob.solution.get_status()

print "corresponding string to the solution status"
print _prob.solution.status[ _prob.solution.get_status() ]

print "Solution value"
print _prob.solution.get_objective_value()

weights = _prob.solution.get_values()
print "weights of assets:"
print weights