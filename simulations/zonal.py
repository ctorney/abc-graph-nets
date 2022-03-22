import numpy as np
from math import *
import sys
import random
import time

from tqdm import tqdm
import tensorflow as tf
from scipy import stats

SPEED = 3.0
ETA = 0.9
NOISE = 0.1
class zonal_model:
    def __init__(self, N, timesteps, discard, repeat, L, dt, save_interval, disable_progress=False):
        self.N = N
        self.timesteps = timesteps
        self.discard = discard
        self.B = repeat  # repeat for B batches
        self.L = L
        self.dt = dt
        self.save_interval = save_interval
        self.macro_state1 = np.zeros((self.B, (self.timesteps - self.discard)//self.save_interval))
        self.macro_state2 = np.zeros((self.B, (self.timesteps - self.discard)//self.save_interval))
        self.macro_state3 = np.zeros((self.B, (self.timesteps - self.discard)//self.save_interval))
        
        # turn progress bar on or off
        self.disable_progress = disable_progress

    def initialise_state(self):

        self.angles = tf.random.uniform((self.B,self.N,1), 0, 2*pi) #
        self.positions = tf.random.uniform((self.B,self.N,2), 0.0, self.L, dtype=tf.float32) #0,self.L)
        angles = tf.random.uniform((self.B,self.N,1), 0, 2*pi,dtype=tf.float32) #

        cos_A = tf.math.cos(angles)
        sin_A = tf.math.sin(angles)

        self.velocities = tf.concat([cos_A,sin_A],axis=-1)
        


    def run_sim(self, *params):

        Rr, Ro, Ra, va = params
        
        
        # tensorflow function to run an update step
        @tf.function
        def update_tf(X, V):
            
            angles = tf.expand_dims(tf.math.atan2(V[...,1],V[...,0]),-1)
            cos_A = tf.math.cos(angles)
            sin_A = tf.math.sin(angles)


            Xx = tf.expand_dims(X[...,0],-1)
            dx = -Xx + tf.linalg.matrix_transpose(Xx)
            dx = tf.where(dx>0.5*self.L, dx-self.L, dx)
            dx = tf.where(dx<-0.5*self.L, dx+self.L, dx)

            Xy = tf.expand_dims(X[...,1],-1)
            dy = -Xy + tf.linalg.matrix_transpose(Xy)
            dy = tf.where(dy>0.5*self.L, dy-self.L, dy)
            dy = tf.where(dy<-0.5*self.L, dy+self.L, dy)


            angle_to_neigh = tf.math.atan2(dy, dx)
            cos_N = tf.math.cos(angle_to_neigh)
            sin_N = tf.math.sin(angle_to_neigh)
            rel_angle_to_neigh = angle_to_neigh - angles
            rel_angle_to_neigh = tf.math.atan2(tf.math.sin(rel_angle_to_neigh), tf.math.cos(rel_angle_to_neigh))
            
            dist = tf.math.sqrt(tf.square(dx)+tf.square(dy))
    
            # repulsion 
            rep_x = tf.where(dist<=Rr, -dx, tf.zeros_like(dx))
            rep_x = tf.where(rel_angle_to_neigh<0.5*va, rep_x, tf.zeros_like(rep_x))
            rep_x = tf.where(rel_angle_to_neigh>-0.5*va, rep_x, tf.zeros_like(rep_x))
            #rep_x = tf.math.divide_no_nan(rep_x,tf.math.square(dist))
            rep_x = tf.reduce_sum(rep_x,axis=2)

            rep_y = tf.where(dist<=Rr, -dy, tf.zeros_like(dy))
            rep_y = tf.where(rel_angle_to_neigh<0.5*va, rep_y, tf.zeros_like(rep_y))
            rep_y = tf.where(rel_angle_to_neigh>-0.5*va, rep_y, tf.zeros_like(rep_y))
            #rep_y = tf.math.divide_no_nan(rep_y,tf.math.square(dist))
            rep_y = tf.reduce_sum(rep_y,axis=2)

            rep_norm = tf.math.sqrt(rep_x**2+rep_y**2)
            rep_x = tf.math.divide_no_nan(rep_x,rep_norm)
            rep_y = tf.math.divide_no_nan(rep_y,rep_norm)

            # alignment 
            align_x = tf.where(dist<=Ro, cos_A, tf.zeros_like(cos_A))
            #align_x = tf.where(dist>Rr, align_x, tf.zeros_like(align_x))

            align_x = tf.where(rel_angle_to_neigh<0.5*va, align_x, tf.zeros_like(align_x))
            align_x = tf.where(rel_angle_to_neigh>-0.5*va, align_x, tf.zeros_like(align_x))
            align_x = tf.reduce_sum(align_x,axis=1)
            
            align_y = tf.where(dist<=Ro, sin_A, tf.zeros_like(sin_A))
            #align_y = tf.where(dist>Rr, align_y, tf.zeros_like(align_y))

            align_y = tf.where(rel_angle_to_neigh<0.5*va, align_y, tf.zeros_like(align_y))
            align_y = tf.where(rel_angle_to_neigh>-0.5*va, align_y, tf.zeros_like(align_y))
            align_y = tf.reduce_sum(align_y,axis=1)

            al_norm = tf.math.sqrt(align_x**2+align_y**2)
            align_x = tf.math.divide_no_nan(align_x,al_norm)
            align_y = tf.math.divide_no_nan(align_y,al_norm)

            # attractive interactions
            attr_x = tf.where(dist<=Ra, dx, tf.zeros_like(dx))
            #attr_x = tf.where(dist>(Ro+Rr), attr_x, tf.zeros_like(attr_x))
            attr_x = tf.where(rel_angle_to_neigh<0.5*va, attr_x, tf.zeros_like(attr_x))
            attr_x = tf.where(rel_angle_to_neigh>-0.5*va, attr_x, tf.zeros_like(attr_x))
            attr_x = tf.reduce_sum(attr_x,axis=2)

            attr_y = tf.where(dist<=Ra, dy, tf.zeros_like(dy))
            #attr_y = tf.where(dist>(Ro+Rr), attr_y, tf.zeros_like(attr_y))
            attr_y = tf.where(rel_angle_to_neigh<0.5*va, attr_y, tf.zeros_like(attr_y))
            attr_y = tf.where(rel_angle_to_neigh>-0.5*va, attr_y, tf.zeros_like(attr_y))
            attr_y = tf.reduce_sum(attr_y,axis=2)

            at_norm = tf.math.sqrt(attr_x**2+attr_y**2)
            attr_x = tf.math.divide_no_nan(attr_x,at_norm)
            attr_y = tf.math.divide_no_nan(attr_y,at_norm)

            # combine angles and convert to desired angle change
            social_x = tf.where(rep_norm>1e-6,rep_x, align_x + attr_x)
            social_y = tf.where(rep_norm>1e-6,rep_y, align_y + attr_y)
            
            # combine angles and convert to desired angle change
            #social_x = rep_x + align_x + attr_x
            #social_y = rep_y + align_y + attr_y
            
            social_norm = tf.math.sqrt(social_x**2+social_y**2)
            social_x = tf.math.divide_no_nan(social_x,social_norm)
            social_y = tf.math.divide_no_nan(social_y,social_norm)

            #d_angle = tf.math.atan2(social_y,social_x)
            #d_angle = social_y + social_x

            social_x = tf.expand_dims(social_x,-1)
            social_y = tf.expand_dims(social_y,-1)

            nvx = (1-ETA)*social_x + ETA*cos_A
            nvy = (1-ETA)*social_y + ETA*sin_A

            v_norm = tf.math.sqrt(nvx**2+nvy**2)
            nvx = tf.math.divide_no_nan(nvx,v_norm)
            nvy = tf.math.divide_no_nan(nvy,v_norm)

            # update velocity
            V = self.dt*SPEED*tf.concat([nvx,nvy],axis=-1)
            
            # update positions
            X += V

            X = tf.where(X>self.L, X-self.L, X)
            X = tf.where(X<0, X+self.L, X)

            X = tf.where(X>self.L, X-self.L, X)
            X = tf.where(X<0, X+self.L, X)

            return X, V
            
            
        self.initialise_state()

        counter=0
        for i in tqdm(range(self.timesteps),disable=self.disable_progress):
            positions, velocities = update_tf(self.positions,  self.velocities)
            self.positions = positions
            self.velocities = velocities
            if i>=self.discard:
                if i%self.save_interval==0:
                        
                    self.macro_state1[:,counter] = self.compute_macro_state1().numpy()
                    self.macro_state2[:,counter] = self.compute_macro_state2().numpy()
                    self.macro_state3[:,counter] = self.compute_macro_state3().numpy()

                    counter = counter + 1
                        
        return 
    

    def compute_macro_state1(self):
        # return the order parameter for each batch

        av_velocity = tf.reduce_mean(self.velocities,axis=1)
        order_parameter = tf.norm(av_velocity,axis=1)  
        
        return order_parameter
    
    def compute_macro_state2(self):
        # return the rotation parameter for each batch
        X = self.positions

        velocity = self.velocities
        c_group = tf.reduce_mean(X,axis=1,keepdims=True) 

        r_ic = X - c_group
        # normalize the vectors (missing in the Couzin paper but required)
        r_ic = r_ic/tf.expand_dims(tf.linalg.norm(r_ic,axis=-1),-1)

        r_ic = tf.concat([r_ic,tf.zeros([self.B,self.N,1])],2)
        velocity = tf.concat([velocity,tf.zeros([self.B,self.N,1])],2)

        r_cross_v = tf.linalg.cross(r_ic,velocity) 

        rotation_parameter = tf.norm(tf.reduce_mean(r_cross_v,axis=1),axis=1)
        
        return rotation_parameter
    
    

        
    def compute_macro_state3(self):
        # return the nearest neighbour distance
        X = self.positions

        Xx = tf.expand_dims(X[...,0],-1)
        dx = -Xx + tf.linalg.matrix_transpose(Xx)
        dx = tf.where(dx>0.5*self.L, dx-self.L, dx)
        dx = tf.where(dx<-0.5*self.L, dx+self.L, dx)

        Xy = tf.expand_dims(X[...,1],-1)
        dy = -Xy + tf.linalg.matrix_transpose(Xy)
        dy = tf.where(dy>0.5*self.L, dy-self.L, dy)
        dy = tf.where(dy<-0.5*self.L, dy+self.L, dy)

        dist = tf.math.sqrt(tf.square(dx)+tf.square(dy))
        dist = tf.where(dist==0.0,tf.zeros_like(dist)+10000.0,dist) #remove individual itself
        #return average nearest neighbour distance:
        return tf.reduce_mean(tf.reduce_min(dist,axis=-1),axis=-1) 
    


    def get_macro_states(self):
    
        order = np.ravel(self.macro_state1[:,:])
        rotation = np.ravel(self.macro_state2[:,:])   # np.ravel(np.diff(self.macro_state))
        nn_dist = np.ravel(self.macro_state3[:,:]) 
        
        return order, rotation, nn_dist
