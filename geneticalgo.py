from scipy.optimize import differential_evolution
# from scipy.optimize import LinearConstraint

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

                        #function for constraints (ONLY FOR CENTRALISED FLEXIBLE CASES)
# counter = 0
# for i in expansionChange:
#     if i == 1:
#         counter += 1

# def nadia(help):
#     return help[0] + (counter*help[1]) - 65882.50

# def nadia(help):
#     return -(help[0] + (3*help[1]) - 65882.50)

#constraints for phased and flexible cases only
# cons = ({'type': 'eq', 'fun': lambda help:  help[0]+(3*help[1])-65882.50})
# cons = {'type': 'eq', 'fun': lambda help: help[0] + (counter*help[1]) - 65882.50}
# cons = ({'type': 'ineq', 'fun': nadia})

# cons = ({'type': 'ineq', 'fun': nadia})

# cons = ({'type': 'eq', 'fun': lambda help:  help[0]+(105*help[1])-65882.50})
# cons = ({'type': 'eq', 'fun': lambda help:  help[0]+((modulesbuiltscum - 35)*help[1])-65882.50})


# Define the bounds
bounds = [(100000, 200000)]                                                #centralised Fixed Certain and Uncertain
# bounds = [(30000, 50000), (0, 1)]                                         #centralised Phased Certain and Uncertain
# bounds = [(30000, 50000), (0, 1), (0, 1)]                              #centralised Flexible Certain and Uncertain

# bounds = [(100, 600)]                                                #decentralised Fixed Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1)]                                     #decentralised Phased Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1), (0, 5), (0, 1)]                            #decentralised Flexible Certain and Uncertain

if __name__ == "__main__":
    result = differential_evolution(test, bounds, workers=-1)
    print("Optimal solution: ", result.x)
    print("Objective function value: ", result.fun)
