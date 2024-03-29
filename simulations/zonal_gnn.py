import numpy as np
from math import *
import sys
import random
import time

from tqdm import tqdm
import tensorflow as tf

import os


SPEED = 3.0
ETA = 0.9
NOISE = 0.1

def get_record(group_id,timestep,parameter_vector,pos,vel,acc):
    feature = { 'group_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[group_id])),
                'timestep': tf.train.Feature(int64_list=tf.train.Int64List(value=[timestep])),
                'parameter_vector': tf.train.Feature(float_list=tf.train.FloatList(value=parameter_vector)),
                'pos': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pos.numpy()])),
                'vel': tf.train.Feature(bytes_list=tf.train.BytesList(value=[vel.numpy()])),
                'acc': tf.train.Feature(bytes_list=tf.train.BytesList(value=[acc.numpy()]))
               
                }
    return tf.train.Example(features=tf.train.Features(feature=feature))


class zonal_model:
    def __init__(self, N, timesteps, discard, repeat, L, dt, save_interval,train_directory='train_datasets', valid_directory='valid_datasets', disable_progress=False, save_micro=False):
        self.N = N
        self.timesteps = timesteps
        self.discard = discard
        self.B = repeat  # repeat for B batches
        self.L = L
        self.dt = dt
        self.save_interval = save_interval
        
        self.micro_state = np.zeros((self.B, (self.timesteps - self.discard)//self.save_interval, N, 6),dtype=np.float32)

        self.sim_counter=0

        

        # turn progress bar on or off
        self.disable_progress = disable_progress

        self.valid_fraction = 0.1
        
        # flag to turn on saving data to tfrecords
        self.save_micro = save_micro
        
        if self.save_micro: 
            if not os.path.exists(train_directory):
                os.makedirs(train_directory)

            if not os.path.exists(valid_directory):
                os.makedirs(valid_directory)

            self.train_directory = train_directory
            self.valid_directory = valid_directory
        
    def initialise_state(self):

        #self.positions = tf.random.uniform((self.B,self.N,2), 0.0, self.L, dtype=tf.float32) #0,self.L)
        self.positions = tf.random.uniform((self.B,self.N,2), 0.5*self.L - 10.0, 0.5*self.L+10.0, dtype=tf.float32) #0,self.L)
        #self.positions = tf.random.uniform((self.B,self.N,2),0.5*self.L, 0.5*self.L+20) #0,self.L)
        #self.positions = tf.random.uniform((self.B,self.N,2),0, self.L) 
        angles = tf.random.uniform((self.B,self.N,1), 0, 2*pi,dtype=tf.float32) #

        cos_A = tf.math.cos(angles)
        sin_A = tf.math.sin(angles)

        self.velocities = tf.concat([cos_A,sin_A],axis=-1)
        

        


    def run_sim(self, *params):

        Rr, Ro, Ra, va = params
        #Ro = Rr + Ror # absolute interaction distance (values are passed in as the interaction zone width)
        #Ra = Ro + Rar # absolute interaction distance (values are passed in as the interaction zone width)
        
        if self.save_micro: 
            record_file = self.train_directory + '/microstates-' + str(self.sim_counter) + '.tfrecords'
            self.writer = tf.io.TFRecordWriter(record_file) 

            valid_file = self.valid_directory + '/microstates-' + str(self.sim_counter) + '.tfrecords'
            self.validwriter = tf.io.TFRecordWriter(valid_file) 
        
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
            #attr_x = tf.where(dist>Ro, attr_x, tf.zeros_like(attr_x))
            attr_x = tf.where(rel_angle_to_neigh<0.5*va, attr_x, tf.zeros_like(attr_x))
            attr_x = tf.where(rel_angle_to_neigh>-0.5*va, attr_x, tf.zeros_like(attr_x))
            attr_x = tf.reduce_sum(attr_x,axis=2)

            attr_y = tf.where(dist<=Ra, dy, tf.zeros_like(dy))
            #attr_y = tf.where(dist>Ro, attr_y, tf.zeros_like(attr_y))
            attr_y = tf.where(rel_angle_to_neigh<0.5*va, attr_y, tf.zeros_like(attr_y))
            attr_y = tf.where(rel_angle_to_neigh>-0.5*va, attr_y, tf.zeros_like(attr_y))
            attr_y = tf.reduce_sum(attr_y,axis=2)

            at_norm = tf.math.sqrt(attr_x**2+attr_y**2)
            attr_x = tf.math.divide_no_nan(attr_x,at_norm)
            attr_y = tf.math.divide_no_nan(attr_y,at_norm)

            # combine angles and convert to desired angle change
            #social_x = tf.where(rep_norm>1e-6,rep_x, align_x + attr_x)
            #social_y = tf.where(rep_norm>1e-6,rep_y, align_y + attr_y)
            
            # combine angles and convert to desired angle change
            social_x = rep_x + align_x + attr_x
            social_y = rep_y + align_y + attr_y
            
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

            #d_angle = tf.math.atan2((1-ETA)*tf.math.sin(d_angle) + ETA*sin_A, (1-ETA)*tf.math.cos(d_angle) + ETA*cos_A)

            # update velocity
            V = self.dt*SPEED*tf.concat([nvx,nvy],axis=-1)
#             d_angle = tf.math.atan2(social_y,social_x)
#             d_angle = tf.expand_dims(d_angle,-1)

            
#             d_angle = tf.math.atan2((1-ETA)*tf.math.sin(d_angle) + ETA*sin_A, (1-ETA)*tf.math.cos(d_angle) + ETA*cos_A)

#             #d_angle = d_angle - angles
#             #d_angle = tf.where(d_angle>pi, d_angle-2*pi, d_angle)
#             #d_angle = tf.where(d_angle<-pi, d_angle+2*pi, d_angle)


#             # add perception noise
#             #noise = tf.random.normal(shape=(self.B,self.N,1),mean=0,stddev=NOISE*(self.dt**0.5))
#             #d_angle = d_angle + noise
            
#             # restrict to maximum turning angle
#             #d_angle = tf.where(tf.math.abs(d_angle)>eta*self.dt, tf.math.sign(d_angle)*eta*self.dt, d_angle)
            
#             # rotate headings
#             #angles = angles + d_angle
            
#             # update acceleration 
#             A = tf.concat([tf.cos(d_angle),tf.sin(d_angle)],axis=-1)
            
#             # update velocity
#             V = self.dt*SPEED*tf.concat([tf.cos(d_angle),tf.sin(d_angle)],axis=-1)
            
            # update positions
            X += V

            # add periodic boundary conditions
            #A = tf.where(A<-pi,  A+2*pi, A)
            #A = tf.where(A>pi, A-2*pi, A)

            X = tf.where(X>self.L, X-self.L, X)
            X = tf.where(X<0, X+self.L, X)

            X = tf.where(X>self.L, X-self.L, X)
            X = tf.where(X<0, X+self.L, X)

            return X, V
            
        self.initialise_state()

        counter=0
        for i in tqdm(range(self.timesteps),disable=self.disable_progress):
            positions, velocities = update_tf(self.positions,  self.velocities)
            if i>=self.discard:
                if i%self.save_interval==0:
                    # store in an array in case we want to visualise
                    self.micro_state[:,counter,:,0:2] = self.positions.numpy()
                    self.micro_state[:,counter,:,2:4] = self.velocities.numpy()
                    self.micro_state[:,counter,:,4:6] = velocities.numpy()#self.accelerations.numpy()
                        
                    if self.save_micro: 
                        self.save_tf_record(counter, params)
                    counter = counter + 1
                        
            self.positions = positions
            self.velocities = velocities
        if self.save_micro: 
            self.writer.close()
            self.validwriter.close()
        self.sim_counter+=1
        return 

    def save_tf_record(self, counter, params):

        

        for b in range(self.B):
            pos =  tf.io.serialize_tensor(self.micro_state[b,counter,:,0:2])
            vel =  tf.io.serialize_tensor(self.micro_state[b,counter,:,2:4])
            acc =  tf.io.serialize_tensor(self.micro_state[b,counter,:,4:6])

            tf_record = get_record(b,counter,params,pos,vel,acc)
            if b> self.B*self.valid_fraction:
                self.writer.write(tf_record.SerializeToString())
            else:
                self.validwriter.write(tf_record.SerializeToString())

        
        return 
    
