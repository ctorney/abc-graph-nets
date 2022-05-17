from math import *
import numpy as np


# parameters to scan
sobol_listx = np.array([0.0,2.0,9.0,14.0])
sobol_listy = np.array([14.0,12.0,5.0,0.0])

num_reps = 10

# simulation parameters 
L= 100 
N= 100 
data_repeat = 100
sim_repeat = 500
discard = 2000
timesteps = 1
save_interval=1
dt=0.1 
va=1.5*pi
lrep= 1  

# emulator parameters
n_wave = 10
n_points = 25 
T = 3
ndim = 2
p_start = np.array([0.0,0.0])
p_range = np.array([25.0,25.0]) 


# sampler parameters

prior = np.array(((0.0,25.0),(0.0,25.0)))  
burnin = 10000 
mcmcsteps = 2000 
thin=10
