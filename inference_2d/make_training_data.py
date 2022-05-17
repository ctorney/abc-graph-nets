import os, sys
import numpy as np
from math import *
from tqdm import tqdm

import tensorflow as tf


sys.path.append('..')
from simulations import zonal_gnn


from scipy.stats import qmc


INPUT_DIM = 2
N_POINTS = 2000


REP = 1.0
VA = 1.5*pi

params_max = np.array([25.0,25.0])
sampler = qmc.Sobol(d=INPUT_DIM, scramble=False)
sampler.fast_forward(1) # skip the first point at origin
sobel_points = sampler.random(N_POINTS)
param_values = np.zeros_like(sobel_points)


#param_values[:,0]= sobel_points[:,0]*(params_max[0]-REP)
#param_values[:,1]= sobel_points[:,1]*(params_max[1] - param_values[:,0] )
param_values[:,0]= sobel_points[:,0]*(params_max[0])
param_values[:,1]= sobel_points[:,1]*(params_max[1])
#param_values[:,1]= sobel_points[:,1]*params_max[1]



L= 100
N= 100 
repeat = 100
discard = 2000
timesteps = 1000
save_interval=100 
dt=0.1 


sim = zonal_gnn.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=repeat, dt=dt,save_interval=save_interval,save_micro=True, disable_progress=True)
    

def evaluate_zonal_model(X):
    sim.run_sim(REP, X[0], X[1], VA)
    return

for i in tqdm(range(param_values.shape[0])):
    evaluate_zonal_model(param_values[i])
                    
