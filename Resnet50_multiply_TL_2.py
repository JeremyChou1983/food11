#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.utils import plot_model
from IPython.display import Image
import matplotlib.pyplot as plt


def res_identity(x, filters):
    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    #first block 
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)

    #second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)


    # third block activation used after adding the input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
     # x = Activation(activations.relu)(x)
        
    # add the input 
    x = x + x_skip
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)
    return x
    
def res_conv(x, s, filters): 
    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters
    
    # First block
    h1 = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    h1 = BatchNormalization()(h1)
    h1 = keras.activations.relu(h1)

    # Second block

    h2_1 = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_1 = BatchNormalization()(h2_1)
    h2_1 = keras.activations.relu(h2_1)
    h2_2 = tf.keras.layers.GlobalAveragePooling2D()(h2_1)
    h2_2 = layers.Dense(f2,activation='sigmoid')(h2_2)
    h2_2 = tf.reshape(h2_2, [-1, 1, 1, f2])
    multi = tf.multiply(h2_1, h2_2)
    
    concat = layers.Concatenate()([h1, multi])

    # Third block
    x = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(concat)
    x = BatchNormalization()(x)

    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add 
    x = x + x_skip + multi
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)
    return x



def resnet50_multiply(output_V3):

  # 1st stage
  # here we perform maxpooling, see the figure above
    x = layers.Conv2D(64,kernel_size=3,strides=1, padding='same',kernel_initializer='he_normal', use_bias=False)(output_V3)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

  # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    
  # ends with average pooling and dense connection

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(11, activation='softmax', kernel_initializer='he_normal')(x)

    return outputs