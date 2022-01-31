import os, sys
import numpy as np
from math import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf


sys.path.append('..')
from simulations import zonal_gnn

from sobol_seq import i4_sobol_generate


INPUT_DIM = 2
N_POINTS = 2000


REP = 1.0
VA = 1.5*pi

params_max = np.array([25.0,25.0])
sobel_points = i4_sobol_generate(INPUT_DIM,N_POINTS)
param_values = np.zeros_like(sobel_points)


param_values[:,0]=REP + sobel_points[:,0]*(params_max[0]-REP)
param_values[:,1]=REP + sobel_points[:,1]*(params_max[1]-REP)



L= 500
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
                    
