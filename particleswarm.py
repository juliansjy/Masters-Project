# from psopy import minimize
# import numpy as np
# import pyswarms as ps
from psopy import minimize
from psopy import init_feasible


from centralised.centralFixed import *
# from centralised.centralPhased import *
# from centralised.centralFlexible import *
# from centralised.centralFixedUncertain import *
# from centralised.centralPhasedUncertain import *
# from centralised.centralFlexibleUncertain import *

# from decentralised.decentralFixed import * 
# from decentralised.decentralPhased import *
# from decentralised.decentralFlexible import *
# from decentralised.decentralFixedUncertain import *
# from decentralised.decentralPhasedUncertain import *
# from decentralised.decentralFlexibleUncertain import *

# constraints = ({'type': 'eq', 'fun': lambda abcde: min(abcde[0], abcde[1]) - min(0, abcde[1])})

# Define the bounds

# Create bounds
# max_bound = 60000 * np.ones(2)
# min_bound = 0
# bounds = (min_bound, max_bound)

# Define the bounds
bounds = [(100000, 200000)]                                                #centralised Fixed Certain and Uncertain
# bounds = [(30000, 50000), (0, 1)]                                         #centralised Phased Certain and Uncertain
# bounds = [(30000, 50000), (0, 1), (0, 1)]                              #centralised Flexible Certain and Uncertain

# bounds = [(100, 600)]                                                #decentralised Fixed Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1)]                                     #decentralised Phased Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1), (0, 5), (0, 1)]                            #decentralised Flexible Certain and Uncertain

# x0 = np.random.uniform(0, 2, (1000, 5))

# Set-up hyperparameters
# options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=bounds)

x0 = init_feasible(constraints=None, low=0., high=2., shape=(1000, 2))
res = minimize(test, x0, options={'g_rate': 1., 'l_rate': 1., 'max_velocity': 4., 'stable_iter': 50})

# Perform optimization
result = minimize(test, iters=1000)

# Define the genetic algorithm
# result = minimize(test, x0)

# # Print the results
# print("Optimal solution: ", result.x)
# print("Objective function value: ", result.fun)

if __name__ == "__main__":
    ...  # Prepare all the arguments
    # result = scipy.optimize.differential_evolution(minimize_me, bounds=function_bounds, args=extraargs,
    #                                                disp=True, polish=False, updating='deferred', workers=-1)
    result = minimize(test, bounds, workers=-1)
    print("Optimal solution: ", result.x)
    print("Objective function value: ", result.fun)