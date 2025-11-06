import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class SelfAttention(layers.Layer):
    
    def __init__(self, attention_units=64, return_sequences=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.attention_units = attention_units
        self.return_sequences = return_sequences
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.attention_units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.attention_units,),
            initializer="zeros",
            trainable=True
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(self.attention_units, 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        ui = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        attention_weights = tf.tensordot(ui, self.u, axes=1)
        
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        attended_features = inputs * attention_weights
        
        if self.return_sequences:
            return attended_features
        else:
            return tf.reduce_sum(attended_features, axis=1)
    
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return input_shape
        else:
            return (input_shape[0], input_shape[2])
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'attention_units': self.attention_units,
            'return_sequences': self.return_sequences
        })
        return config


class BahdanauAttention(layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        query_expanded = tf.expand_dims(query, 1)
        
        score = self.V(tf.nn.tanh(self.W1(query_expanded) + self.W2(values)))
        
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector = attention_weights * values
        
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        return config
