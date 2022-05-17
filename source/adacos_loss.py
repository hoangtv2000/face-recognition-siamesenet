import math
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import math

from keras import regularizers
import tensorflow_probability as tfp




class AdaCos(tf.keras.layers.Layer):
  def __init__(self, n_classes=5, m=0.50, regularizer=regularizers.l2(1e-1), **kwargs):
    super(AdaCos, self).__init__(**kwargs)
    self.n_classes = n_classes
    self.s = tf.Variable(initial_value=tf.constant(math.sqrt(2)*math.log(n_classes-1)), dtype = tf.float32,trainable=False,aggregation=tf.VariableAggregation.MEAN)
    self.m = m
    self.regularizer = regularizers.get(regularizer)
    self.pi = math.pi


  def build(self, input_shape):
    super(AdaCos, self).build(input_shape[0])
    self.W = self.add_weight(name='W',
                            shape=(input_shape[0][-1], self.n_classes),
                            initializer='glorot_uniform',
                            trainable=True,
                            regularizer=self.regularizer)
    
  def call(self, inputs):
    x, y = inputs
    # normalize feature
    x = tf.nn.l2_normalize(x, axis=1)
    # normalize weights
    W = tf.nn.l2_normalize(self.W, axis=0)
    # dot product
    logits = x @ W
    
    # add margin
    # clip logits to prevent zero division when backward
    theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))

    B_avg = tf.where(y < 1, tf.exp(self.s*logits), tf.zeros_like(logits))
    B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
    theta_class = tf.gather(theta, tf.math.argmax(tf.cast(y, tf.int32),axis=1), name='theta_class')
    theta_med = tfp.stats.percentile(theta_class, q=50)

    with tf.control_dependencies([theta_med, B_avg]):
      temp_s = tf.cast(tf.math.log(B_avg)  / tf.cos(tf.minimum(self.pi/4, theta_med)),tf.float32)
      self.s.assign(temp_s)
      logits = self.s * logits 
      out = tf.nn.softmax(logits)
    return out

  def compute_output_shape(self, input_shape):
    return (None, self.n_classes)




def calculate_adacos_logits(embds, labels, one_hot, embedding_size, class_num,
                            is_dynamic=True):
    weights = tf.get_variable(name='final_dense',
                              shape=[embedding_size, class_num],
                              dtype=tf.float32,
                              initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                              trainable=True)

    init_s = math.sqrt(2) * math.log(class_num - 1)
    adacos_s = tf.get_variable(name='adacos_s_value', dtype=tf.float32,
                              initializer=tf.constant(init_s),
                              trainable=False,
                              aggregation=tf.VariableAggregation.MEAN)
                              
    embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)

    logits_before_s = tf.matmul(embds, weights, name='adacos_logits_before_s')

    if is_dynamic == False:
        output = tf.multiply(init_s, logits_before_s, name='adacos_fixed_logits')
        return output

    theta = tf.acos(tf.clip_by_value(logits_before_s, -1.0 + 1e-10, 1.0 - 1e-10))
      
    B_avg = tf.where_v2(tf.less(one_hot, 1),
                        tf.exp(adacos_s*logits_before_s), tf.zeros_like(logits_before_s))
    B_avg = tf.reduce_mean(tf.reduce_sum(B_avg, axis=1), name='B_avg')
    idxs = tf.squeeze(labels)
    theta_class = tf.gather_nd(theta, tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1),
                              name='theta_class')
    theta_med = tf.contrib.distributions.percentile(theta_class, q=50)
    
    with tf.control_dependencies([theta_med, B_avg]):
        temp_s = tf.log(B_avg) / tf.cos(tf.minimum(math.pi/4, theta_med))
        adacos_s = tf.assign(adacos_s, temp_s)
        output = tf.multiply(adacos_s, logits_before_s, name='adacos_dynamic_logits')
        
    return output