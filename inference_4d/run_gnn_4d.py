#GNN 4D over data instantiations

import threading
import numpy as np

import time
np.set_printoptions(suppress=True,precision=3)

import os, sys
from math import *
import tensorflow as tf 

import scipy

sys.path.append('..')


from gpabc import gp_abc
from gpabc import am_sampler
from simulations import zonal_gnn
from gnn_model import model
from params4d import *


def setup_and_run_hmc(threadid):
    np.random.seed(threadid)
    tf.random.set_seed(threadid)


    
    lali = sobol_listx[threadid]
    latt = sobol_listy[threadid]
    va = sobol_listva[threadid]
    lrep = sobol_listlrep[threadid]

        
    for data_rep in range(num_reps):  
    
        gnn_model = tf.keras.models.load_model('gnn/gnn_model')
        data_sim = zonal_gnn.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=data_repeat, dt=dt,save_interval=save_interval,disable_progress=True)
        
        
        
        data_sim.run_sim(lrep, lali, latt, va)
    
        
        data_sum_stats = []
    
        for i in range(data_repeat):
        
            X = data_sim.micro_state[i,:,:,:2]
            V = data_sim.micro_state[i,:,:,2:4]
            A = data_sim.micro_state[i,:,:,4:]
            
            
            data_sum_stats.append(gnn_model(model.parse_graph([X,V,A])[0]).numpy())
         
        
        macrodata = np.array(data_sum_stats).reshape((-1,2))
        
        
        data_vector = np.mean(macrodata,axis=0) 
        sd0 =  np.std(macrodata,axis=0) 
        cov = np.diag(sd0**2)
        def abc_likelihood_4d(sim_output):
            repeat = sim_output.shape[0]
            k = sim_output.shape[1]
                                    
            return np.log(1e-18 + 1/repeat * (((2*pi*np.product(sd0))**k)**0.5)*np.sum(scipy.stats.multivariate_normal(data_vector,cov).pdf(sim_output)))

        
        
        def simulator_4d(params):
            
            sim = zonal_gnn.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=sim_repeat, dt=dt,save_interval=save_interval,disable_progress=True)
            sim.run_sim(params[0], params[1], params[2], params[3])
            sum_stats = []
        
            for i in range(sim_repeat):
        
                X = sim.micro_state[i,:,:,:2]
                V = sim.micro_state[i,:,:,2:4]
                A = sim.micro_state[i,:,:,4:]
        
        
                sum_stats.append(gnn_model(model.parse_graph([X,V,A])[0]).numpy())#[:,1:3])
        
            return np.array(sum_stats).reshape((-1,2))    
        
        #4D inference:
        
        data_lali = lali
        data_latt = latt
        
        abcGP = gp_abc.abcGP(p_start,p_range,ndim,n_points,T,simulator_4d,abc_likelihood_4d) 
        
        for i in range(n_wave):
            abcGP.runWave()     
            abcGP.remove_implausible()
    
            
        #am_sampler:
        Y = abcGP.sobol_points[np.isfinite(abcGP.likelihood)]
        logl = abcGP.predict_final(Y)[0]
        startval = Y[np.argsort(-logl[:,0])[0]]
        print(startval)
        
        # step size is 1/50th of the plausible range
        steps = np.ptp(abcGP.sobol_points,axis=0)/100
        import time
        start = time.time()
        samples = am_sampler.am_sampler(abcGP.predict_final,ndim,startval,prior,steps, n_samples=mcmcsteps, burn_in=burnin, m=thin)
        print(time.time()-start)        
        
    
        filename = 'results/4d_gnn/rep_' + str(data_rep) + '_DI_' + str(threadid) + '.npy'
        np.save(filename,samples)






def parallel_run(threadid, gpu):
    with tf.name_scope(gpu):
        with tf.device(gpu):
            setup_and_run_hmc(threadid)
    return



gpu_list = tf.config.experimental.list_logical_devices('GPU')
num_threads = len(gpu_list)

print(num_threads)
threads = list()
start = time.time()
for index in range(num_threads):
    x = threading.Thread(target=parallel_run, args=(index,gpu_list[index].name))
    threads.append(x)
    x.start()

for index, thread in enumerate(threads):
    thread.join()

end = time.time()
print('Threaded time taken: ', end-start)    
