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


# In[2]:


# Resnet 50


# In[1]:


def res_identity(x, filters): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added
    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

  #first block 
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)

  #second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)
  

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

    x_skip = x
    f1, f2 = filters


  # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
  # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)

  # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)

  #third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

  # shortcut 
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x_skip)
    x_skip = BatchNormalization()(x_skip)

  # add 
    x = x + x_skip
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)
    return x

def resnet50():

    inputs = keras.Input(shape=(64, 64, 3))
    x = layers.ZeroPadding2D(padding=(3, 3))(inputs)

  # 1st stage
  # here we perform maxpooling, see the figure above

    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

  # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

  # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    
  # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    
  # ends with average pooling and dense connection

    x = layers.AveragePooling2D((2, 2), padding='same')(x)

    x = layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(11, activation='softmax', kernel_initializer='he_normal')(x)

  # define the model 

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Resnet50')

    return model

