#Change the imported case and bounds for that case's design variables
#Workers=-1 forces all CPU cores to run the optimisation

from scipy.optimize import differential_evolution

#Imported cases
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

#Bounds for design variables
bounds = [(100000, 200000)]                                                 #centralised Fixed Certain and Uncertain
# bounds = [(30000, 50000), (0, 1)]                                         #centralised Phased Certain and Uncertain
# bounds = [(30000, 50000), (0, 1), (0, 1)]                                 #centralised Flexible Certain and Uncertain

# bounds = [(100, 600)]                                                     #decentralised Fixed Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1)]                                    #decentralised Phased Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1), (0, 5), (0, 1)]                    #decentralised Flexible Certain and Uncertain

if __name__ == "__main__":
    result = differential_evolution(main, bounds, workers=-1)                #Deterministic cases use "npv", for uncertainty cases use "enpv"
    print("Optimal solution: ", result.x)
    print("Objective function value: ", result.fun)
