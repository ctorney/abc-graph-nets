import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Lambda, Dropout
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from spektral.layers import ECCConv, GlobalAvgPool, MessagePassing, XENetConv, GlobalAttentionPool, GlobalMaxPool, GlobalSumPool,GlobalAttnSumPool


DOMAIN_SIZE=100.
MAX_RADIUS=250.


def parse_graph(inputs, targets=None):
    
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

    output_x = node_velocities

    return (output_x, output_a, output_e, output_i,output_ie), targets
    


def get_gnn_model(n_outputs): #, max_values):

    n_feat_node=2
    n_feat_edge=5

    MLP_SIZE=16*2

    X_in = Input(shape=(n_feat_node,))
    A_in = Input(shape=(None,), sparse=True)
    E_in = Input(shape=(n_feat_edge,))
    I_in = Input(shape=(), dtype=tf.int64)
    IE_in = Input(shape=(), dtype=tf.int64)


    X = Dense(MLP_SIZE, activation="relu")(X_in)
    E = Dense(MLP_SIZE, activation="relu")(E_in)
    #X = Dropout(0.5)(X)


    X, E = XENetConv([MLP_SIZE,MLP_SIZE], MLP_SIZE, 2*MLP_SIZE, node_activation="tanh", edge_activation="tanh")([X, A_in, E])
    X, E = XENetConv([MLP_SIZE,MLP_SIZE], MLP_SIZE, 2*MLP_SIZE, node_activation="tanh", edge_activation="tanh")([X, A_in, E])

    X = Dropout(0.1)(X)
    X = Dense(MLP_SIZE, activation="relu",use_bias=True)(X)

    Xm = GlobalMaxPool()([X, I_in])
    Xa = GlobalAvgPool()([X, I_in])

    X = Concatenate()([Xm,Xa])
    X = Dropout(0.1)(X)

    X = Dense(MLP_SIZE, activation="relu",use_bias=True)(X)
    X = Dropout(0.1)(X)

    output = Dense(n_outputs, activation="softplus",use_bias=True)(X)
    #output = Lambda(lambda X: tf.math.multiply(X, tf.constant(max_values,dtype=tf.float32)))(X) 

    gnn_model = Model(inputs=[X_in, A_in, E_in, I_in, IE_in], outputs=output)

    return gnn_model

