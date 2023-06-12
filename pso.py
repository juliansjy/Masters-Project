import random

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
bounds = [(100000, 200000)]                                                             #centralised Fixed Certain and Uncertain
# bounds = [(30000, 50000), (0, 1)]                                                     #centralised Phased Certain and Uncertain
# bounds = [(30000, 50000), (0, 1), (0, 1)]                                             #centralised Flexible Certain and Uncertain

# bounds = [(100, 600)]                                                                 #decentralised Fixed Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1)]                                                #decentralised Phased Certain and Uncertain
# bounds = [(100, 600), (0, 50), (0, 1), (0, 5), (0, 1)]                                #decentralised Flexible Certain and Uncertain

# ------------------------------------------------------------------------------
# TO CUSTOMIZE THIS PSO CODE TO SOLVE UNCONSTRAINED OPTIMIZATION PROBLEMS, CHANGE THE PARAMETERS IN THIS SECTION ONLY:
# THE FOLLOWING PARAMETERS MUST BE CHANGED.
nv = 1  # number of variables
mm = -1  # if minimization problem, mm = -1; if maximization problem, mm = 1

# THE FOLLOWING PARAMETERS ARE OPTIONAL
particle_size = 50  # number of particles
iterations = 100  # max number of iterations
w = 0.75  # inertia constant
c1 = 1  # cognative constant
c2 = 2  # social constant
# END OF THE CUSTOMIZATION SECTION
# ------------------------------------------------------------------------------

class Particle:
    def __init__(self, bounds):
        self.particle_position = []
        self.particle_velocity = []
        self.local_best_particle_position = []
        self.fitness_local_best_particle_position = initial_fitness
        self.fitness_particle_position = initial_fitness

        for i in range(nv):
            self.particle_position.append(random.uniform(bounds[i][0], bounds[i][1]))
            self.particle_velocity.append(random.uniform(-1, 1))

    def evaluate(self, test):
        self.fitness_particle_position = test(self.particle_position)

        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position
                self.fitness_local_best_particle_position = self.fitness_particle_position
        
        if mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position = self.particle_position
                self.fitness_local_best_particle_position = self.fitness_particle_position

    def update_velocity(self, global_best_particle_position):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()

            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity
    
    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]

            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]

            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]

if mm == -1:
    initial_fitness = float("inf")  # for minimization problem
if mm == 1:
    initial_fitness = -float("inf")  # for maximization problem

fitness_global_best_particle_position = initial_fitness
global_best_particle_position = []
swarm_particle = []

for i in range(particle_size):
    swarm_particle.append(Particle(bounds))
A = []

for i in range(iterations):
    for j in range(particle_size):
        swarm_particle[j].evaluate(main)

        if mm == -1:
            if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                global_best_particle_position = list(swarm_particle[j].particle_position)
                fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
        
        if mm == 1:
            if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                global_best_particle_position = list(swarm_particle[j].particle_position)
                fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)

    for j in range(particle_size):
        swarm_particle[j].update_velocity(global_best_particle_position)
        swarm_particle[j].update_position(bounds)

    A.append(fitness_global_best_particle_position)

print("The best solution is: ", global_best_particle_position)
print("The best fitness for the best solution is: ", fitness_global_best_particle_position)