import os, sys
import numpy as np
from math import *
from tqdm import tqdm
import matplotlib.pyplot as plt

import tensorflow as tf

from scipy import stats

import pickle



from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from spektral.layers import ECCConv, GlobalAvgPool, MessagePassing, XENetConv, GlobalAttentionPool, GlobalMaxPool, GlobalSumPool,GlobalAttnSumPool

plt.style.use('ggplot')
plt.style.use('seaborn-paper') 
plt.style.use('seaborn-whitegrid') 



#************************************
#************************************
#********DATA LOADER*****************
#************************************
#************************************
#************************************

train_dir = 'train_datasets/'
valid_dir = 'valid_datasets/'

BATCH_SIZE=64
EPOCHS=200

all_file_list = [train_dir + filename for filename in os.listdir(train_dir)]

dataset_size = sum(1 for _ in tf.data.TFRecordDataset(all_file_list[0]))*len(all_file_list)//BATCH_SIZE


feature_description = {'group_id': tf.io.FixedLenFeature([], tf.int64),
                        'timestep': tf.io.FixedLenFeature([], tf.int64),
                        'parameter_vector': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
                        'pos': tf.io.FixedLenFeature([], tf.string),
                        'vel': tf.io.FixedLenFeature([], tf.string),
                        'acc': tf.io.FixedLenFeature([], tf.string)}

def _parse_record(x):
  # Parse the input tf.train.Example proto using the dictionary above.
  return tf.io.parse_single_example(x, feature_description)

def _parse_tensor(x):
    output = {'group_id': x['group_id'],
                'timestep': x['timestep'],
                'parameter_vector': x['parameter_vector'],
                'pos': tf.io.parse_tensor(x['pos'],out_type=tf.float32),
                'vel': tf.io.parse_tensor(x['vel'],out_type=tf.float32),
                'acc': tf.io.parse_tensor(x['acc'],out_type=tf.float32)}
    return output

def _parse_keras(x):
    pos = x['pos']
    #pos.set_shape((None,2))
    vel = x['vel']
    #vel.set_shape((None,2))
    acc = x['acc']
    #acc.set_shape((None,2))
    
    target = x['parameter_vector'][1:3]
    #target.set_shape((4))
    output = ((pos,vel,acc),target)
    return output


DOMAIN_SIZE=500.
MAX_RADIUS=25.

max_params = np.array([5.0,25.0,25.0,2*np.pi],dtype=np.float32)

def _parse_graph(inputs, targets):
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

    return (output_x, output_a, output_e, output_i,output_ie), targets/max_params
    
    
    

train_dataset =  tf.data.TFRecordDataset(tf.data.Dataset.list_files([train_dir + filename for filename in os.listdir(train_dir)]))

parsed_train_dataset = train_dataset.map(_parse_record)
parsed_train_dataset = parsed_train_dataset.map(_parse_tensor)
parsed_train_dataset = parsed_train_dataset.map(_parse_keras)


parsed_train_dataset = parsed_train_dataset.shuffle(10000, reshuffle_each_iteration=True)
parsed_train_dataset = parsed_train_dataset.repeat(EPOCHS)
parsed_train_dataset = parsed_train_dataset.batch(BATCH_SIZE, drop_remainder=True)

parsed_train_dataset = parsed_train_dataset.map(_parse_graph)


valid_dataset =  tf.data.TFRecordDataset(tf.data.Dataset.list_files([valid_dir + filename for filename in os.listdir(valid_dir)]))

parsed_valid_dataset = valid_dataset.map(_parse_record)
parsed_valid_dataset = parsed_valid_dataset.map(_parse_tensor)
parsed_valid_dataset = parsed_valid_dataset.map(_parse_keras)
parsed_valid_dataset = parsed_valid_dataset.batch(BATCH_SIZE, drop_remainder=True)
parsed_valid_dataset = parsed_valid_dataset.map(_parse_graph)



strategy = tf.distribute.MirroredStrategy()



#************************************
#************************************
#********GNN MODEL*******************
#************************************
#************************************
#************************************

n_out = 2
n_feat_node=4
n_feat_edge=5

MLP_SIZE=32

#with strategy.scope():
if True:

    X_in = Input(shape=(n_feat_node,))
    A_in = Input(shape=(None,), sparse=True)
    E_in = Input(shape=(n_feat_edge,))
    I_in = Input(shape=(), dtype=tf.int64)
    IE_in = Input(shape=(), dtype=tf.int64)



    X = Dense(MLP_SIZE, activation="linear")(X_in)
    E = Dense(MLP_SIZE, activation="linear")(E_in)


    X, E = XENetConv([MLP_SIZE,MLP_SIZE], MLP_SIZE, 2*MLP_SIZE, node_activation="tanh", edge_activation="tanh")([X, A_in, E])
    X, E = XENetConv([MLP_SIZE,MLP_SIZE], MLP_SIZE, 2*MLP_SIZE, node_activation="tanh", edge_activation="tanh")([X, A_in, E])

    X = Dense(MLP_SIZE, activation="linear",use_bias=False)(X)
#E = Dense(MLP_SIZE, activation="linear",use_bias=False)(E)


    X = Concatenate()([X, X_in])
#E = Concatenate()([E, E_in])

    Xs = GlobalAttnSumPool()([X, I_in])
    Xm = GlobalMaxPool()([X, I_in])
    Xa = GlobalAvgPool()([X, I_in])

#Es = GlobalAttnSumPool()([E, IE_in])
#Em = GlobalMaxPool()([E, IE_in])
#Ea = GlobalAvgPool()([E, IE_in])

    X = Concatenate()([Xs,Xm,Xa])#, Es,Em,Ea])

    X = Dense(MLP_SIZE, activation="linear",use_bias=False)(X)

    output = Dense(n_out, activation="sigmoid",use_bias=False)(X)

    gnn_model = Model(inputs=[X_in, A_in, E_in, I_in, IE_in], outputs=output)


    learning_rate = 1e-3# Learning rate
    gnn_model.compile(optimizer=Adam(learning_rate), loss="mse")

if not os.path.exists('gnn'):
    os.makedirs('gnn')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='gnn/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                                            save_weights_only=True,
                                                            monitor='val_loss',
                                                            mode='max',
                                                            save_best_only=True)

stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                    min_delta=0,
                                                    patience=5,
                                                    verbose=0,
                                                    mode="auto",
                                                    restore_best_weights=True)

gnn_model.fit(parsed_train_dataset, steps_per_epoch=dataset_size, epochs=EPOCHS, validation_data=parsed_valid_dataset, callbacks=[checkpoint_callback,stop_callback])

gnn_model.save('gnn/gnn_model')


#************************************
#************************************
#********SAVE FIGURE*****************
#************************************
#************************************
#************************************




pred_list = []
true_values = []
for databatch in tqdm(parsed_valid_dataset):

    target = databatch[1]
    true_values.append(target.numpy())


    predictions = gnn_model(databatch[0])
    pred_list.append(np.squeeze(predictions.numpy()))





fig, axs = plt.subplots(1,2, figsize=(8, 3), facecolor='w', edgecolor='k')  

axs = axs.ravel()
for pred_i in range(4):

    pred_vals = np.array([pp[:,pred_i] for pp in pred_list]).flatten()
    true_vals = np.array([tt[:,pred_i] for tt in true_values]).flatten()

    bin_means, bin_edges, binnumber = stats.binned_statistic(true_vals, pred_vals,bins=100)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    bin_stds, bin_edges, binnumber = stats.binned_statistic(true_vals, pred_vals,statistic='std',bins=100)


    axs[pred_i].plot(bin_centers,bin_means,c='C0')

    axs[pred_i].fill_between(bin_centers,bin_means-bin_stds,bin_means+bin_stds,color='C0',alpha=0.5)

    xx = np.linspace(0,true_vals.max(),10)
    axs[pred_i].plot(xx,xx,c='k',ls='--')

    axs[pred_i].set_ylabel('GNN prediction of parameter')
    axs[pred_i].set_xlabel('True parameter that generated the microstate')



plt.savefig('gnn_trained_4d.png',dpi=300)
