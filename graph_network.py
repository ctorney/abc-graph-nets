
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
  return snt.nets.MLP(output_sizes=[hidden_size] * num_hidden_layers + [output_size],activation=activation,activate_final=activate_final)


class EncodeProcessDecode(snt.Module):
  """Encode-Process-Decode function approximator for learnable simulator."""

  def __init__(
      self,
      latent_size: int,
      mlp_hidden_size: int,
      mlp_num_hidden_layers: int,
      num_message_passing_steps: int,
      output_size: int,
      output_min: np.array,
      output_max: np.array,
      reducer: Reducer = tf.math.unsorted_segment_sum,
      name: str = "EncodeProcessDecode"):
    """Inits the model.

    Args:
      latent_size: Size of the node and edge latent representations.
      mlp_hidden_size: Hidden layer size for all MLPs.
      mlp_num_hidden_layers: Number of hidden layers in all MLPs.
      num_message_passing_steps: Number of message passing steps.
      output_size: Output size of the decode node representations as required
        by the downstream update function.
      reducer: Reduction to be used when aggregating the edges in the nodes in
        the interaction network. This should be a callable whose signature
        matches tf.math.unsorted_segment_sum.
      name: Name of the model.
    """

    super().__init__(name=name)

    self._latent_size = latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._output_size = output_size
    self._output_min = tf.convert_to_tensor(output_min)
    self._output_range = tf.convert_to_tensor(output_max - output_min)
    self._reducer = reducer

    #with self._enter_variable_scope():
    self._networks_builder()

  def __call__(self, input_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Forward pass of the learnable dynamics model."""

    # Encode the input_graph.
    latent_graph_0 = self._encode(input_graph)

    # Do `m` message passing steps in the latent graphs.
    latent_graph_m = self._process(latent_graph_0)
    decode_graph_m = self._decode(latent_graph_m)
    # Decode from the last latent graph.
    return self._output_min + self._output_range*tf.nn.sigmoid(self._output_transform(decode_graph_m.globals))

  def _networks_builder(self):
    """Builds the networks."""
    def build_mlp_with_layer_norm():
      mlp = build_mlp(
          hidden_size=self._mlp_hidden_size,
          num_hidden_layers=self._mlp_num_hidden_layers,
          output_size=self._latent_size)
      return snt.Sequential([mlp, snt.LayerNorm(axis=slice(1, None),create_scale=False,create_offset=False)])
    

    # The encoder graph network independently encodes edge and node features.
    encoder_kwargs = dict(
        edge_model_fn=build_mlp_with_layer_norm,
        node_model_fn=build_mlp_with_layer_norm)
    self._encoder_network = gn.modules.GraphIndependent(**encoder_kwargs)
    # Create `num_message_passing_steps` graph networks with unshared parameters
    # that update the node and edge latent features.
    # Note that we can use `modules.InteractionNetwork` because
    # it also outputs the messages as updated edge latent features.
    self._processor_networks = []
    for _ in range(self._num_message_passing_steps):
        #encoder_kwargs = dict(
        #    edge_model_fn=build_mp_with_layer_norm,
        #    node_model_fn=build_mp_with_layer_norm,reducer=self._reducer)
        self._processor_networks.append(
              gn.modules.InteractionNetwork(edge_model_fn=build_mlp_with_layer_norm,
                  node_model_fn=build_mlp_with_layer_norm,
                  reducer=self._reducer))
    # The decoder MLP decodes node latent features into the output size.
    
    decoder_kwargs = dict(
    global_model_fn=build_mlp_with_layer_norm)
    self._decoder_network = gn.modules.GraphIndependent(**decoder_kwargs)
    
    #self._output_transform =tf.keras.models.Sequential()

    #for _ in range(self._mlp_num_hidden_layers):
    #    self._output_transform.add(tf.keras.layers.Dense(self._mlp_hidden_size, activation='relu'))

    #self._output_transform.add(tf.keras.layers.Dense(self._output_size,activation=tf.nn.softplus))

    self._output_transform = build_mlp(
        hidden_size=self._mlp_hidden_size,
        num_hidden_layers=self._mlp_num_hidden_layers,
        output_size=self._output_size) #, activation=tf.nn.softplus,activate_final=True)
  def _encode(
      self, input_graph: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Encodes the input graph features into a latent graph."""

    ## Copy the globals to all of the nodes, if applicable.
    #if input_graph.globals is not None:
    #  broadcasted_globals = gn.blocks.broadcast_globals_to_nodes(input_graph)
    #  input_graph = input_graph.replace(
    #      nodes=tf.concat([input_graph.nodes, broadcasted_globals], axis=-1),
    #      globals=None)

    # Encode the node and edge features.
    latent_graph_0 = self._encoder_network(input_graph)
    return latent_graph_0

  def _process(
      self, latent_graph_0: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Processes the latent graph with several steps of message passing."""

    # Do `m` message passing steps in the latent graphs.
    # (In the shared parameters case, just reuse the same `processor_network`)
    latent_graph_prev_k = latent_graph_0
    latent_graph_k = latent_graph_0
    for processor_network_k in self._processor_networks:
      latent_graph_k = self._process_step(
          processor_network_k, latent_graph_prev_k)
      latent_graph_prev_k = latent_graph_k

    latent_graph_m = latent_graph_k
    reducer = tf.math.unsorted_segment_sum

    latent_graph_m = latent_graph_m.replace(globals=gn.blocks.NodesToGlobalsAggregator(reducer=reducer)(latent_graph_m))
    #num_graphs = utils_tf.get_num_graphs(latent_graph_m)
    #graph_index = tf.range(num_graphs)
    #indices = utils_tf.repeat(graph_index, latent_graph_m.n_node, axis=0)#,
                              #sum_repeats_hint=_get_static_num_nodes(graph))
        
    #indices = tf.repeat(graph_index, latent_graph_m.n_node,axis=0)
    #new_globals = tf.math.unsorted_segment_sum(latent_graph_m.nodes, indices, num_graphs)
    #latent_graph_m = latent_graph_m.replace(globals=new_globals)

    return latent_graph_m

  def _process_step(
      self, processor_network_k: snt.Module,
      latent_graph_prev_k: gn.graphs.GraphsTuple) -> gn.graphs.GraphsTuple:
    """Single step of message passing with node/edge residual connections."""

    # One step of message passing.
    latent_graph_k = processor_network_k(latent_graph_prev_k)

    # Add residuals.
    latent_graph_k = latent_graph_k.replace(
        nodes=latent_graph_k.nodes+latent_graph_prev_k.nodes,
        edges=latent_graph_k.edges+latent_graph_prev_k.edges)
    return latent_graph_k

  def _decode(self, latent_graph: gn.graphs.GraphsTuple) -> tf.Tensor:
    """Decodes from the latent graph."""
    return self._decoder_network(latent_graph)

