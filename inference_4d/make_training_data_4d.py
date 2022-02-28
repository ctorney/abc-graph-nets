import os, sys
import numpy as np
from math import *
from tqdm import tqdm

import tensorflow as tf


sys.path.append('..')
from simulations import zonal_gnn


from scipy.stats import qmc


INPUT_DIM = 4
N_POINTS = 2000


#REP = 1.0
#VA = 1.5*pi

params_max = np.array([5.0,25.0,25.0,2*np.pi]) #l_ali, l_att, l_rep, v_a
sampler = qmc.Sobol(d=INPUT_DIM, scramble=False)
sampler.fast_forward(1) # skip the first point at origin
sobel_points = sampler.random(N_POINTS)
param_values = np.zeros_like(sobel_points)


param_values[:,0]=sobel_points[:,0]*params_max[0]
param_values[:,3]=sobel_points[:,3]*params_max[3]
param_values[:,1]=sobel_points[:,0] + sobel_points[:,1]*(params_max[1]-param_values[:,0])
param_values[:,2]=sobel_points[:,0] + sobel_points[:,2]*(params_max[2]-param_values[:,0])

#param_values[:,0]=sobel_points[:,0]*params_max[0]
#param_values[:,3]=sobel_points[:,3]*params_max[3]
#param_values[:,1]=sobel_points[:,1]*params_max[1] 
#param_values[:,2]=sobel_points[:,2]*params_max[2]


L= 500
N= 100 
repeat = 100
discard = 2000
timesteps = 1000
save_interval=100 
dt=0.1 


sim = zonal_gnn.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=repeat, dt=dt,save_interval=save_interval,save_micro=True, disable_progress=True)
    

def evaluate_zonal_model(X):
    sim.run_sim(X[0], X[1], X[2], X[3])
#    sim.run_sim(X[2], X[0], X[1], X[3])
    return

for i in tqdm(range(param_values.shape[0])):
    evaluate_zonal_model(param_values[i])