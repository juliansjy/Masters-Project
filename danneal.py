from scipy.optimize import dual_annealing
# import numpy as np

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
bounds = [(100000, 200000)]                                                #centralised Fixed Certain and Uncertain
# bounds = [(30000, 50000), (0, 1)]                                         #centralised Phased Certain and Uncertain
# bounds = [(30000, 50000), (0, 1), (0, 1)]                              #centralised Flexible Certain and Uncertain

# bounds = [(100, 600)]                                                #decentralised Fixed Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1)]                                     #decentralised Phased Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1), (0, 5), (0, 1)]                            #decentralised Flexible Certain and Uncertain


# Define the genetic algorithm
result = dual_annealing(test, bounds)

# Print the results
print("Optimal solution: ", result.x)
print("Objective function value: ", result.fun)

# if __name__ == "__main__":
#     ...  # Prepare all the arguments
#     # result = scipy.optimize.differential_evolution(minimize_me, bounds=function_bounds, args=extraargs,
#     #                                                disp=True, polish=False, updating='deferred', workers=-1)
#     result = dual_annealing(test, bounds, workers=-1)
#     print("Optimal solution: ", result.x)
#     print("Objective function value: ", result.fun)
