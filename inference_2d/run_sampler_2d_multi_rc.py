#running with matched noise models, and new multiscale am sampler.

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
#from gabc import am_sampler_multi

#from sobol_seq import i4_sobol_generate
from scipy.stats import qmc
from scipy.stats import gaussian_kde

from simulations import zonal

#sobol_points = np.load('sobol_points_2d.npy')
    
##
#sobol_listx = np.array([5,7.5,2.5,3.75]) #0,3,5,12])   
#sobol_listy = np.array([5,2.5,7.5,3.75]) #15,15,13,9]) 
sobol_listx = np.array([1,3,10,20]) #0,3,5,12])   
sobol_listy = np.array([15,15,10,10]) #15,15,13,9]) 



   # observability_sim = obs_list[threadid]
# specify observation values to use here


def setup_and_run_hmc(threadid):
    np.random.seed(threadid)
    tf.random.set_seed(threadid)


    num_reps = 10
    burnin = 5000 #10000
    mcmcsteps = 1000 #8000
 
    
    
    #for dl in range(di):
        
    lali = sobol_listx[threadid]
    latt = sobol_listy[threadid]
#        lali = sobol_points[dl,0]  #data instantiation
#        latt = sobol_points[dl,1] 
        
    for data_rep in range(num_reps):             
        ## generate data
        L= 500 #200
        discard= 2000 #2500 #5000 
        N= 100 #500
        repeat = 100 #20 #100#0 
        timesteps = 2 
        save_interval=1 #0 #1
        dt=0.1 #1 #0.1
        simulation_cls = zonal.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=repeat, dt=dt,save_interval=save_interval)
            
            
        lrep= 1
        eta=0.9
        va=1.5*pi
        vs=3 #5
        sigma=0.1
        simulation_cls.run_sim(eta, latt, lali, lrep, vs, va, sigma)
            
        data_eta=eta
        data_va=va
        data_latt=latt
        data_lali=lali
        data_lrep=lrep
        data_vs = vs
        data_sigma = sigma
            
        DATA_y = [data_lali,data_latt,data_lrep,data_eta,data_vs,data_va,data_sigma]
            
            
        op, rot, ent, nnd, dis = simulation_cls.get_macro_states()
        avgOPDATA=np.zeros(repeat)
        avgROTDATA=np.zeros(repeat)
        avgENTDATA=np.zeros(repeat)
        avgNNDDATA=np.zeros(repeat)
        avgDISDATA=np.zeros(repeat)
            
            
        for i in range(repeat):
            avgOPDATA[i] = op[((i+1)*(timesteps-1))-1]  
            avgROTDATA[i] = rot[((i+1)*(timesteps-1))-1] 
            avgENTDATA[i] = ent[((i+1)*(timesteps-1))-1] 
            avgNNDDATA[i] = nnd[((i+1)*(timesteps-1))-1] 
            avgDISDATA[i] = dis[((i+1)*(timesteps-1))-1] 
            
        macrodata=  np.array([avgOPDATA,avgROTDATA,avgENTDATA,avgNNDDATA,avgDISDATA])
        macrodata = np.squeeze([macrodata[np.r_[0:2,3],None]]) 
            

            
        def abc_likelihood_2d(sim_output,rc):
            theta_0 = rc@sim_output

            ss_0 = rc@macrodata

            theta_DATA0 = np.mean(ss_0,axis=-1) 
            sd0 =  np.std(ss_0,axis=-1) 
            cov = np.diag(sd0**2)
            repeat = sim_output.shape[1]
            
            k = sd0.shape[0]
            return np.log(1e-18 + 1/repeat * (((2*pi)**k)**0.5*np.product(sd0))*np.sum(scipy.stats.multivariate_normal(theta_DATA0,cov).pdf(theta_0.T)))


        def simulator_2d(params):
            repeat = 50    
            simulation_cls = zonal.zonal_model(N,timesteps+discard,discard=discard,repeat=repeat,L=L,dt=dt, save_interval=1,disable_progress=True) 

            simulation_cls.run_sim(eta, params[1], params[0],lrep, vs, va, sigma) 
            
            output = np.array(simulation_cls.get_macro_states()) 
            
            return np.squeeze([output[np.r_[0:2,3],None]]) 
#            output = simulation_cls.get_macro_states() 
#            
#            return np.array(output)
                            
        #2D inference of l_ali and eta: 
        ndim = 2
        p_start = np.array([0.0,0.0])
        p_range = np.array([25.0,25.0]) 
                
        # use values for plotting the predicted GP
        X = np.array([np.linspace(p_start[0],p_start[0]+p_range[0],100),np.linspace(p_start[1],p_start[1]+p_range[1],100)])
        y_previous = np.full((100,100),np.log(1e-18))
                
        # number of waves
        n_wave = 10 
        n_points = 20 
        T = 3
                

        abcGP = gp_abc.abcGP(p_start,p_range,ndim,n_points,T,simulator_2d,abc_likelihood_2d) 
                
                
        for i in range(n_wave):
            abcGP.runWave()
            abcGP.remove_implausible()                
            abcGP.update_rc()
                

              
        #sampling:
        # random plausible point to start with
        startval = abcGP.sobel_points[np.random.choice(abcGP.sobel_points.shape[0])]
        # step size is 1/50th of the plausible range
        steps = np.ptp(abcGP.sobel_points,axis=0)/50
        prior = np.array(((0.0,25.0),(0.0,25.0)))
        samples = am_sampler.am_sampler(abcGP.predict_final,2,startval,prior, steps,n_samples=mcmcsteps, burn_in=burnin, m=100)


        filename = '2d_multi_rc/SR_rep_' + str(data_rep) + '_DI_' + str(threadid) + '.npy'
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