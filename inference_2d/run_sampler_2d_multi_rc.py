
import threading
import numpy as np

import time
np.set_printoptions(suppress=True,precision=3)

import os, sys
from math import *
import tensorflow as tf 


sys.path.append('..')


from gpabc import gp_abc
from gpabc import am_sampler
from simulations import zonal
from params2d import *




def setup_and_run_hmc(threadid):
    np.random.seed(threadid)
    tf.random.set_seed(threadid)

        
    lali = sobol_listx[threadid]
    latt = sobol_listy[threadid]
        
    for data_rep in range(num_reps):             
        simulation_cls = zonal.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=data_repeat, dt=dt,save_interval=save_interval)
            
        simulation_cls.run_sim(lrep, lali, latt, va)
            
        data_va=va
        data_latt=latt
        data_lali=lali
        data_lrep=lrep
            
        op, rot, nnd = simulation_cls.get_macro_states()
        macrodata=  np.array([op[:,-1], rot[:,-1],nnd[:,-1]])

            
        def abc_likelihood_2d(sim_output,rc):
            theta_0 = rc@sim_output

            ss_0 = rc@macrodata
            theta_DATA0 = np.mean(ss_0,axis=-1) 
            sd0 =  np.std(ss_0,axis=-1) 
            cov = np.diag(sd0**2)
            repeat = sim_output.shape[1]
            
            k = sd0.shape[0]
            #return np.log(1e-18 + 1/repeat * (((2*pi*np.product(sd0))**k)**0.5)*np.sum(scipy.stats.multivariate_normal(theta_DATA0,cov).pdf(theta_0.T)))
            return np.log(1e-18 + 1/repeat * (((2*pi)**k)**0.5*np.product(sd0))*np.sum(scipy.stats.multivariate_normal(theta_DATA0,cov).pdf(theta_0.T)))


        def simulator_2d(params):
            simulation_cls = zonal.zonal_model(N,timesteps+discard,discard=discard,repeat=sim_repeat,L=L,dt=dt, save_interval=1,disable_progress=True) 
            simulation_cls.run_sim(lrep, params[0], params[1], va)
            output = np.array(simulation_cls.get_macro_states()) 
            return output[...,-1] 
                            

        abcGP = gp_abc_rc.abcGP(p_start,p_range,ndim,n_points,T,simulator_2d,abc_likelihood_2d) 
                
                
        for i in range(n_wave):
            abcGP.runWave()
            abcGP.remove_implausible()                
            abcGP.update_rc()
                

              
        #am_sampler:
        Y = abcGP.sobol_points[np.isfinite(abcGP.likelihood)]
        logl = abcGP.predict_final(Y)[0]
        startval = Y[np.argsort(-logl[:,0])[0]]
        # step size is 1/50th of the plausible range
        steps = np.ptp(abcGP.sobol_points,axis=0)/50
        samples = am_sampler.am_sampler(abcGP.predict_final,ndim,startval,prior,steps, n_samples=mcmcsteps, burn_in=burnin, m=skip)


        filename = '2d_multi_rc/rep_' + str(data_rep) + '_DI_' + str(threadid) + '.npy'
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
