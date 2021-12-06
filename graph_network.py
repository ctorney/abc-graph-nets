
# graph nets for ABC



# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Graph network implementation based on 
    https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate

"""
from typing import Callable
from graph_nets import utils_tf

import numpy as np
import graph_nets as gn
import sonnet as snt
import tensorflow as tf

Reducer = Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]


def build_mlp(hidden_size: int, num_hidden_layers: int, output_size: int, activation=tf.nn.relu,activate_final=False) -> snt.Module:
    """Builds an MLP."""
    #,w_init = snt.initializers.RandomNormal()
    return snt.nets.MLP(output_sizes=[hidden_size] * num_hidden_layers + [output_size], activation=activation,activate_final=activate_final)


class EncodeProcessDecode(snt.Module):
    """Encode-Process-Decode function approximator for learnable simulator."""

    def __init__(self,latent_size: int, mlp_hidden_size: int, mlp_num_hidden_layers: int, num_message_passing_steps: int, output_size: int, domain_size: float,
                      reducer: Reducer = tf.math.unsorted_segment_sum, name: str = "EncodeProcessDecode"):
        """Initialise the model.

        Args:
          latent_size: Size of the node and edge latent representations.
          mlp_hidden_size: Hidden layer size for all MLPs.
          mlp_num_hidden_layers: Number of hidden layers in all MLPs.
          num_message_passing_steps: Number of message passing steps.
          output_size: Output size of the decode node representations as required by the downstream update function.
          domain_size: Size of the simulation for rescaling positions
          reducer: Reduction to be used when aggregating the edges in the nodes in the interaction network. 
          name: Name of the model.
        """

        super().__init__(name=name)

        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer
        self._L = domain_size

        self._networks_builder()

    def __call__(self, X, V, A) -> tf.Tensor:
        """Forward pass of the learnable dynamics model."""

        # Preprocess the microstate.
        input_graph = self._preprocess_data(X,V,A)

        # Encode the input_graph.
        latent_graph_0 = self._encode(input_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)
        decode_graph_m = self._decode(latent_graph_m)
        # Decode from the last latent graph.
        return tf.nn.softplus(self._output_transform(decode_graph_m.globals))

  
    @tf.function
    def _preprocess_data(self, X, V, A):
    
        # node features xpos, ypos, xvel, yvel
        # edge features distance, rel angle to receiver

        Xx = tf.expand_dims(X[...,0],-1)
        dx = -Xx + tf.linalg.matrix_transpose(Xx)
        dx = tf.where(dx>0.5*self._L, dx-self._L, dx)
        dx = tf.where(dx<-0.5*self._L, dx+self._L, dx)

        Xy = tf.expand_dims(X[...,1],-1)
        dy = -Xy + tf.linalg.matrix_transpose(Xy)
        dy = tf.where(dy>0.5*self._L, dy-self._L, dy)
        dy = tf.where(dy<-0.5*self._L, dy+self._L, dy)

        Vx = tf.expand_dims(V[...,0],-1)
        dvx = -Vx + tf.linalg.matrix_transpose(Vx)

        Vy = tf.expand_dims(V[...,1],-1)
        dvy = -Vy + tf.linalg.matrix_transpose(Vy)

        angles = tf.expand_dims(tf.math.atan2(V[...,1],V[...,0]),-1)
        angle_to_neigh = tf.math.atan2(dy, dx)

        rel_angle_to_neigh = angle_to_neigh - angles

        dist = tf.math.sqrt(tf.square(dx)+tf.square(dy))
        #print(tf.reduce_mean(dist,axis=[1,2]))
        interaction_radius = 25.0# tf.reduce_mean(dist,axis=[1,2],keepdims=True)
        adj_matrix = tf.where(dist<interaction_radius, tf.ones_like(dist,dtype=tf.int32), tf.zeros_like(dist,dtype=tf.int32))
        adj_matrix = tf.linalg.set_diag(adj_matrix, tf.zeros(tf.shape(adj_matrix)[:2],dtype=tf.int32))
        sender_recv_list = tf.where(adj_matrix)
        n_edge = tf.reduce_sum(adj_matrix, axis=[1,2])
        n_node = tf.ones_like(n_edge)*tf.shape(adj_matrix)[-1]
        #g_globals = tf.zeros_like(n_edge,dtype=tf.float32) 

        senders =tf.squeeze(tf.slice(sender_recv_list,(0,1),size=(-1,1)))+ tf.squeeze(tf.slice(sender_recv_list,(0,0),size=(-1,1)))*tf.shape(adj_matrix,out_type=tf.int64)[-1]
        receivers = tf.squeeze(tf.slice(sender_recv_list,(0,2),size=(-1,1))) + tf.squeeze(tf.slice(sender_recv_list,(0,0),size=(-1,1)))*tf.shape(adj_matrix,out_type=tf.int64)[-1]


        edge_distance = tf.expand_dims(tf.gather_nd(dist/self._L,sender_recv_list),-1)
        edge_x_distance =  tf.expand_dims(tf.gather_nd(tf.math.cos(rel_angle_to_neigh),sender_recv_list),-1)  # neigbour position relative to sender heading
        edge_y_distance =  tf.expand_dims(tf.gather_nd(tf.math.sin(rel_angle_to_neigh),sender_recv_list),-1)  # neigbour position relative to sender heading

        edge_x_orientation =  tf.expand_dims(tf.gather_nd(dvx,sender_recv_list),-1)  # neigbour velocity relative to sender heading
        edge_y_orientation =  tf.expand_dims(tf.gather_nd(dvy,sender_recv_list),-1)  # neigbour velocity relative to sender heading


        edges = tf.concat([edge_distance,edge_x_distance,edge_y_distance,edge_x_orientation,edge_y_orientation],axis=-1)
        #edges = tf.concat([edge_distance,edge_x_distance,edge_y_distance],axis=-1)

        node_positions = tf.reshape(X,(-1,2))
        node_positions = (node_positions - (self._L/2.))/self._L
        node_velocities = tf.reshape(V,(-1,2))
        node_accelerations = tf.reshape(A,(-1,2))

        nodes = tf.concat([node_positions,node_velocities,node_accelerations],axis=-1)

        input_graphs = gn.graphs.GraphsTuple(nodes=nodes,edges=edges,globals=None,receivers=receivers,senders=senders,n_node=n_node,n_edge=n_edge)

        input_graphs = utils_tf.set_zero_global_features(input_graphs,self._output_size)
        return input_graphs  

    def _networks_builder(self):
        """Builds the networks."""
        def build_mlp_with_layer_norm():
            mlp = build_mlp(hidden_size=self._mlp_hidden_size, num_hidden_layers=self._mlp_num_hidden_layers, output_size=self._latent_size)
            return snt.Sequential([mlp, snt.LayerNorm(axis=slice(1, None),create_scale=False,create_offset=False)])
    

        # The encoder graph network independently encodes edge and node features.
        encoder_kwargs = dict(edge_model_fn=build_mlp_with_layer_norm,node_model_fn=build_mlp_with_layer_norm)
        self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)
        
        # Create `num_message_passing_steps` graph networks with unshared parameters
        # that update the node and edge latent features.
        # Note that we can use `modules.InteractionNetwork` because
        # it also outputs the messages as updated edge latent features.
        self._processor_networks = []
        for _ in range(self._num_message_passing_steps):
            self._processor_networks.append(gn.modules.GraphNetwork(edge_model_fn=build_mlp_with_layer_norm,
                                                                    node_model_fn=build_mlp_with_layer_norm, 
                                                                    global_model_fn=build_mlp_with_layer_norm, 
                                                                    reducer=tf.math.unsorted_segment_mean, 
                                                                    node_block_opt=dict(use_sent_edges=True)))
        
        # The decoder MLP decodes node latent features into the output size.
        decoder_kwargs = dict(global_model_fn=build_mlp_with_layer_norm)
        self._decoder_network = gn.modules.GraphIndependent(**decoder_kwargs)

        self._output_transform = build_mlp(hidden_size=self._mlp_hidden_size, num_hidden_layers=self._mlp_num_hidden_layers, output_size=self._output_size) 
  

    def _encode(self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        """Encodes the input graph features into a latent graph."""

        # Encode the node and edge features.
        latent_graph_0 = self._encoder_network(input_graph)
        return latent_graph_0

    def _process(self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        """Processes the latent graph with several steps of message passing."""

        # Do `m` message passing steps in the latent graphs.
        # (In the shared parameters case, just reuse the same `processor_network`)
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(processor_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_k
        #reducer = tf.math.unsorted_segment_mean

        #latent_graph_m = latent_graph_m.replace(globals=gn.blocks.NodesToGlobalsAggregator(reducer=reducer)(latent_graph_m))

        return latent_graph_m

    def _process_step(self, processor_network_k: snt.Module, latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
        """Single step of message passing with node/edge residual connections."""

        # One step of message passing.
        latent_graph_k = processor_network_k(latent_graph_prev_k)

        # Add residuals.
        latent_graph_k = latent_graph_k.replace(nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
                                                edges=latent_graph_k.edges+latent_graph_prev_k.edges)
        return latent_graph_k

    def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
        """Decodes from the latent graph."""
        return self._decoder_network(latent_graph)

