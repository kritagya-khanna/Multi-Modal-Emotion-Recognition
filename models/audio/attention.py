import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class SelfAttention(layers.Layer):
    """Self-attention mechanism for sequence data"""
    
    def __init__(self, attention_units=64, return_sequences=False, **kwargs):
        """
        Initialize self-attention layer
        
        Args:
            attention_units: Number of attention units
            return_sequences: Whether to return the full sequence or just the attended features
        """
        super(SelfAttention, self).__init__(**kwargs)
        self.attention_units = attention_units
        self.return_sequences = return_sequences
    
    def build(self, input_shape):
        """
        Build layer weights based on input shape
        
        Args:
            input_shape: Shape of input tensor
        """
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
        """
        Apply attention to inputs
        
        Args:
            inputs: Sequence of features (batch_size, time_steps, features)
        
        Returns:
            Attended features
        """
        # Compute attention scores
        # shape: (batch_size, time_steps, attention_units)
        ui = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        
        # Compute attention weights
        # shape: (batch_size, time_steps, 1)
        attention_weights = tf.tensordot(ui, self.u, axes=1)
        
        # Apply softmax to get normalized attention weights
        # shape: (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)
        
        # Apply attention weights to input sequence
        # shape: (batch_size, time_steps, features)
        attended_features = inputs * attention_weights
        
        if self.return_sequences:
            # Return full sequence with attention weights applied
            return attended_features
        else:
            # Sum over time dimension to get context vector
            # shape: (batch_size, features)
            return tf.reduce_sum(attended_features, axis=1)
    
    def compute_output_shape(self, input_shape):
        """
        Compute output shape
        
        Args:
            input_shape: Shape of input tensor
        
        Returns:
            Output shape
        """
        if self.return_sequences:
            return input_shape
        else:
            return (input_shape[0], input_shape[2])
    
    def get_config(self):
        """Get layer configuration"""
        config = super(SelfAttention, self).get_config()
        config.update({
            'attention_units': self.attention_units,
            'return_sequences': self.return_sequences
        })
        return config


class BahdanauAttention(layers.Layer):
    """Bahdanau attention mechanism"""
    
    def __init__(self, units):
        """
        Initialize Bahdanau attention layer
        
        Args:
            units: Number of attention units
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        """
        Apply attention to inputs
        
        Args:
            query: Query tensor, typically the hidden state (batch_size, query_dim)
            values: Sequence of values (batch_size, time_steps, value_dim)
        
        Returns:
            context_vector: Attended context vector (batch_size, value_dim)
            attention_weights: Attention weights (batch_size, time_steps, 1)
        """
        # query: (batch_size, query_dim)
        # values: (batch_size, time_steps, value_dim)
        
        # Expand query to match time dimension of values
        # (batch_size, 1, query_dim)
        query_expanded = tf.expand_dims(query, 1)
        
        # Compute score
        # score: (batch_size, time_steps, 1)
        score = self.V(tf.nn.tanh(self.W1(query_expanded) + self.W2(values)))
        
        # Apply softmax to get attention weights
        # attention_weights: (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply attention weights to values
        # context_vector: (batch_size, time_steps, value_dim)
        context_vector = attention_weights * values
        
        # Sum over time dimension
        # context_vector: (batch_size, value_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
    
    def get_config(self):
        """Get layer configuration"""
        config = super(BahdanauAttention, self).get_config()
        return config