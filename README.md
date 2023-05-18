# Techno-Economic Evaluation and Optimisation of Flexible, Sustainable and Resilient Hydrogen Production Systems Design under Uncertainty
## This is my Final Year Project at Imperial College London Dyson School of Design Engineering
## Used a Macbook Pro 14 inch base model with M2 Pro Apple Silicon Chip

#### Project is about running optimisation algorithms to get the best NPV and ENPV values from both a centralised and decentralised blue hydrogen production and deployment systems
#### Runtimes will be very long so please keep device charged and running for up to 4 days for optimisations to run completely and get similar results

#### Code base uses numpy, matplotlib and SciPy to complete all simulations and optimisations
#### Have these libraries installed before running any code in the terminal

##### To run the files, please go into the correct directories and run "time python3 {file name}"
##### "time python3" will provide the duration that your device took to run the entire simulation and optimisation

##### Deterministic cases can be identified without the "uncertain" at the end of the file name. These files will take very short to run
##### Uncertainty cases can be identified with an "uncertain" at the end of the file name. These files will take at least an hour to run over 2000 iterations
##### Optimisation functions from the SciPy library used are "differential_evolution", "dual_annealing" and "fmin"
##### Particle swarm algorithm (psy.py file) is created manually as it is not available in SciPy. This will utilise only 1 cpu core on any device and will take at least 4 days to run continuously
##### Genetic Algorithm (differential_evolution function) will take between 1-4 hours to run for each case
##### Longest to run: Decentralised Flexible Uncertain (among uncertainty cases)
##### Shortest to run: Centralised Fixed Uncertain (among uncertainty cases)

##### Centralised Folder contains 3 deterministic cases (fixed, phased, flexible) and 3 uncertainty cases (fixed, phased, flexible)
##### Decentralised Folder contains 3 deterministic cases (fixed, phased, flexible) and 3 uncertainty cases (fixed, phased, flexible)
##### Use geneticalgo.py file to run optimisations on all centralised and decentralised cases
##### Runtimes will be quite long and will utilise all cpu cores available. 
##### Particle swarm will also run but will utilise only a single core.
##### You may also try to use the fmin and dual_annealing functions from SciPy to run the optimisations
