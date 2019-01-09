import sys
import cplex
from cplex.exceptions import CplexError

#Maximize
#	x1+2x2+3x3
#subject to
#	-x1+x2+x3<=20
#	x1-3x2+x3<=30
#with these bounds
#	0<=x1<=40
#	0<=x2<=infinity
#	0<=x3<=infinity

# variables
_obj = [1.0 , 2.0 , 3.0]    # _obj are the coefficients of the objective function
_ub = [40.0 , cplex.infinity , cplex.infinity ]
_colnames = ["x1" , "x2" , "x3"]
# constraints
_rhs = [20.0 , 30.0]
_rownames = ["c1" , "c2"]
_sense = "LL"
_rows = [
    [ _colnames , [-1.0 , 1.0 , 1.0] ],
    [ _colnames , [1.0 , -3.0 , 1.0] ]
]

# init solver
_prob = cplex.Cplex()
# define objective sense : maximise or minimise
_prob.objective.set_sense(_prob.objective.sense.maximize)
# define objective variables (names + objective coeffs) and their bounds (upper and lower) # lower bounds default to 0.0
_prob.variables.add( names=_colnames, obj=_obj, ub=_ub )

### Querying the variables
# list of all lower bounds
#lbs = _prob.variables.get_lower_bounds()
# list of upper bounds
#ubs = _prob.variables.get_upper_bounds(0)
# name of the variables
#names = _prob.variables.get_names([0,2])

_prob.linear_constraints.add(lin_expr= _rows , senses=_sense , rhs= _rhs , names= _rownames)

# because there are two arguments, they are taken to specify a range
# thus, cols is the entire constraint matrix as a list of column vectors
#cols = _prob.variables.get_cols("x1", "x3")

_prob.solve()

numrows = _prob.linear_constraints.get_num()
numcols = _prob.variables.get_num()

print "Solution status"
print _prob.solution.get_status()

print "corresponding string to the solution status"
print _prob.solution.status[ _prob.solution.get_status() ]

print "Solution value"
print _prob.solution.get_objective_value()

slack = _prob.solution.get_linear_slacks()
dv = _prob.solution.get_dual_values()
values = _prob.solution.get_values()
rc = _prob.solution.get_reduced_costs()

print "slack"
print slack

print "dual values"
print dv

print "values"
print values

print "reduced costs"
print rc