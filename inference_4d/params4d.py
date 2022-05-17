from math import *
import numpy as np


# parameters to scan
sobol_listx = np.array([1.0,3.0,10.0,10.0])
sobol_listy = np.array([9.0,12.0,10.0,5.0]) 
sobol_listva = np.array([1.5*pi, pi, 0.5*pi, 1.75*pi]) 
sobol_listlrep = np.array([0.5,2.0,3.0,1.0])    

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

# emulator parameters
n_wave = 10
n_points = 25 
T = 3
ndim = 4
p_start = np.array([0.0,0.0,0.0,0.0])
p_range = np.array([5.0,25.0,25.0,2*pi]) 


# sampler parameters

prior = np.array(((0.0,5.0),(0.0,25.0),(0.0,25.0),(0.0,2*pi)))  
burnin = 10000 
mcmcsteps = 2000 
thin=10
