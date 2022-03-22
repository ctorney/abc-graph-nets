#GNN 2D over data instantiations

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

from sobol_seq import i4_sobol_generate
from scipy.stats import gaussian_kde




from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model

from simulations import zonal_gnn
from gpabc import gp_abc
from gpabc import am_sampler

from sobol_seq import i4_sobol_generate
from scipy.stats import gaussian_kde

import pickle

#from graph_network import EncodeProcessDecode

import tensorflow as tf

#sobol_points = np.load('sobol_points_2d.npy')
    
##
sobol_listx = np.array([1.0,3.0,10.0,20.0]) #0,3,5,12])   
sobol_listy = np.array([15.0,15.0,10.0,10.0]) #15,15,13,9]) 

sobol_listva = np.array([1.5*pi, pi, pi/2, 1.75*pi]) 
sobol_listlrep = np.array([0.5,2.0,3.0,1.0])    


   # observability_sim = obs_list[threadid]
# specify observation values to use here


def setup_and_run_hmc(threadid):
    np.random.seed(threadid)
    tf.random.set_seed(threadid)


    num_reps = 10
    burnin = 5000 #1000 #10000 mini test first
    mcmcsteps = 1000 #2000 #8000
    
    lali = sobol_listx[threadid]
    latt = sobol_listy[threadid]
    #eta = sobol_listeta[threadid]
    va = sobol_listva[threadid]
    lrep = sobol_listlrep[threadid]

#        lali = sobol_points[dl,0]  #data instantiation
#        latt = sobol_points[dl,1] 
        
    for data_rep in range(num_reps):  
    
        ####
     
        #max_params = np.array([25.0,25.0],dtype=np.float32)
        MAX_RADIUS=25.
        DOMAIN_SIZE=500.
        
        def _parse_graph(inputs):
            #inputs, targets = x
            X, V, A = inputs
            
            Xx = tf.expand_dims(X[...,0],-1)
            dx = -Xx + tf.linalg.matrix_transpose(Xx)
            dx = tf.where(dx>0.5*DOMAIN_SIZE, dx-DOMAIN_SIZE, dx) 
            dx = tf.where(dx<-0.5*DOMAIN_SIZE, dx+DOMAIN_SIZE, dx) 
        
            Xy = tf.expand_dims(X[...,1],-1)
            dy = -Xy + tf.linalg.matrix_transpose(Xy)
            dy = tf.where(dy>0.5*DOMAIN_SIZE, dy-DOMAIN_SIZE, dy) 
            dy = tf.where(dy<-0.5*DOMAIN_SIZE, dy+DOMAIN_SIZE, dy) 
        
            Vx = tf.expand_dims(V[...,0],-1)
            dvx = -Vx + tf.linalg.matrix_transpose(Vx)
        
            Vy = tf.expand_dims(V[...,1],-1)
            dvy = -Vy + tf.linalg.matrix_transpose(Vy)
            
            dvnorm = tf.math.sqrt(dvx**2+dvy**2)
            dvx = tf.math.divide_no_nan(dvx,dvnorm)
            dvy = tf.math.divide_no_nan(dvy,dvnorm)
        
            angles = tf.expand_dims(tf.math.atan2(V[...,1],V[...,0]),-1)
            angle_to_neigh = tf.math.atan2(dy, dx) 
        
            rel_angle_to_neigh = angle_to_neigh - angles
        
            dist = tf.math.sqrt(tf.square(dx)+tf.square(dy))
        
            adj_matrix = tf.where(dist<MAX_RADIUS, tf.ones_like(dist,dtype=tf.int32), tf.zeros_like(dist,dtype=tf.int32))
            adj_matrix = tf.linalg.set_diag(adj_matrix, tf.zeros(tf.shape(adj_matrix)[:2],dtype=tf.int32))
            sender_recv_list = tf.where(adj_matrix)
            n_edge = tf.reduce_sum(adj_matrix, axis=[1,2])
            n_node = tf.ones_like(n_edge)*tf.shape(adj_matrix)[-1]
        
            output_i = tf.repeat(tf.range(tf.shape(adj_matrix)[0]),n_node)
            output_ie = tf.repeat(tf.range(tf.shape(adj_matrix)[0]),n_edge)
        
        
            senders =tf.squeeze(tf.slice(sender_recv_list,(0,1),size=(-1,1)))+ tf.squeeze(tf.slice(sender_recv_list,(0,0),size=(-1,1)))*tf.shape(adj_matrix,out_type=tf.int64)[-1]
            receivers = tf.squeeze(tf.slice(sender_recv_list,(0,2),size=(-1,1))) + tf.squeeze(tf.slice(sender_recv_list,(0,0),size=(-1,1)))*tf.shape(adj_matrix,out_type=tf.int64)[-1]
        
            output_a = tf.sparse.SparseTensor(indices=tf.stack([senders,receivers],axis=1), values = tf.ones_like(senders),dense_shape=[tf.shape(output_i)[0],tf.shape(output_i)[0]])
            edge_distance = tf.expand_dims(tf.gather_nd(dist/MAX_RADIUS, sender_recv_list),-1)
            edge_x_distance =  tf.expand_dims(tf.gather_nd(tf.math.cos(rel_angle_to_neigh),sender_recv_list),-1)  # neigbour position relative to sender heading
            edge_y_distance =  tf.expand_dims(tf.gather_nd(tf.math.sin(rel_angle_to_neigh),sender_recv_list),-1)  # neigbour position relative to sender heading
        
            edge_x_orientation =  tf.expand_dims(tf.gather_nd(dvx,sender_recv_list),-1)  # neigbour velocity relative to sender heading
            edge_y_orientation =  tf.expand_dims(tf.gather_nd(dvy,sender_recv_list),-1)  # neigbour velocity relative to sender heading
        
        
            output_e = tf.concat([edge_distance,edge_x_distance,edge_y_distance,edge_x_orientation,edge_y_orientation],axis=-1)
        
            node_velocities = tf.reshape(V,(-1,2))
            node_accelerations = tf.reshape(A,(-1,2))
        
            output_x = tf.concat([node_velocities,node_accelerations],axis=-1)
        
            return output_x, output_a, output_e, output_i,output_ie#), targets/max_params    
        
        #####
        gnn_model = tf.keras.models.load_model('gnn/gnn_model')
        
        
        
        #####
        L= 500 #500
        N= 100 
        repeat = 20 #100
        discard = 2500 #2000
        timesteps = 1000
        save_interval=100
        dt=0.1 
        eta = 0.9
        
        data_sim = zonal_gnn.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=repeat, dt=dt,save_interval=save_interval,disable_progress=True)
        
   
        
        
        data_sim.run_sim(lrep, lali, latt, va)
    
        #####
        
        data_sum_stats = []
    
        for i in range(repeat):
        
            X = data_sim.micro_state[i,:,:,:2]
            V = data_sim.micro_state[i,:,:,2:4]
            A = data_sim.micro_state[i,:,:,4:]
            
            
            data_sum_stats.append(gnn_model(_parse_graph([X,V,A])).numpy())#[:,1:3])
         
        
        
        macrodata = np.array(data_sum_stats).reshape((-1,2))
        
        
        
        ####
        data_vector = np.mean(macrodata,axis=0) 
        cov = np.std(macrodata,axis=0).mean()**2
        
        def abc_likelihood_2d(sim_output):
            #ss_0 = macrodata
            #'theta_DATA0 = np.mean(ss_0,axis=0) 
            #'sd0 =  np.std(ss_0,axis=0) 
            #'cov = np.diag(sd0**2)
            repeat = sim_output.shape[0]
            k = sim_output.shape[1]
            
            return np.log(1e-18 + 1/repeat * (((2*pi*cov)**k)**0.5)*np.sum(scipy.stats.multivariate_normal(data_vector,cov).pdf(sim_output)))
        
        
        sim = zonal_gnn.zonal_model(N,timesteps=timesteps+discard,discard=discard,L=L,repeat=repeat, dt=dt,save_interval=save_interval,disable_progress=True, save_micro=True)
        
        def simulator_2d(params):
            #repeat = 50    
            
        
        
            
            sim.run_sim(params[0], params[1], params[2], params[3])
            
            
            sum_stats = []
        
        
            for i in range(repeat):
        
                X = sim.micro_state[i,:,:,:2]
                V = sim.micro_state[i,:,:,2:4]
                A = sim.micro_state[i,:,:,4:]
        
        
                sum_stats.append(gnn_model(_parse_graph([X,V,A])).numpy())#[:,1:3])
        
            return np.array(sum_stats).reshape((-1,2))    
        
        #####
        #4D inference:
        ndim = 4
        p_start = np.array([0.0,0.0,0.0,0.0])
        p_range = np.array([5.0,25.0,25.0,2*pi]) 
        
        # use values for plotting the predicted GP
        #X = np.array([np.linspace(p_start[0],p_start[0]+p_range[0],100),np.linspace(p_start[1],p_start[1]+p_range[1],100)])
        #y_previous = np.full((100,100),np.log(1e-18))
        
        # number of waves
        n_wave = 10
        n_points = 20 
        T = 3
        
        # number of points to add per wave
        #n_points = 80
        #T=0.05
        
        data_lali = lali
        data_latt = latt
        
        abcGP = gp_abc.abcGP(p_start,p_range,ndim,n_points,T,simulator_2d,abc_likelihood_2d) #synth_likelihood_function) #likelihood_function)
        ####
        
        for i in range(n_wave):
            abcGP.runWave()     
            
            abcGP.remove_implausible()
    
            #if i>0:
            #y_pred, y_std  = abcGP.predict_final(x_grid,remove_implausible=True)
    
            
            
        ####
            
        #am_sampler:
        Y = abcGP.sobol_points[np.isfinite(abcGP.likelihood)]
        logl = abcGP.predict_final(Y)[0]
        startval = Y[np.argsort(-logl[:,0])[0]]
        #startval = abcGP.sobel_points[np.random.choice(abcGP.sobel_points.shape[0])]
        prior = np.array(((0.0,5.0),(0.0,25.0),(0.0,25.0),(0.0,2*pi)))  
        print(startval)
        
        # step size is 1/50th of the plausible range
        steps = np.ptp(abcGP.sobol_points,axis=0)/100
        import time
        start = time.time()
        #samples = am_sampler.am_sampler(abcGP.predict_final,2,startval,prior,steps, n_samples=1000, burn_in=5000, m=20)
        samples = am_sampler.am_sampler(abcGP.predict_final,4,startval,prior,steps, n_samples=mcmcsteps, burn_in=burnin, m=20)
        print(time.time()-start)        
        
        ####
        
    
        filename = '4d_gnn/rep_' + str(data_rep) + '_DI_' + str(threadid) + '.npy'
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
