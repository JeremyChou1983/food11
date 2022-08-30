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


# Model_3


# In[7]:


def res_identity(x, filters): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters
    # First block
    h1 = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x)
    h1 = BatchNormalization()(h1)
    h1 = keras.activations.relu(h1)

    # Second block
    h2_1 = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_1 = BatchNormalization()(h2_1)
    h2_1 = keras.activations.relu(h2_1)

    h2_2 = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_2 = BatchNormalization()(h2_2)
    h2_2 = keras.activations.relu(h2_2)

    h2_3 = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_3 = BatchNormalization()(h2_3)
    h2_3 = keras.activations.relu(h2_3)
    h2_3 = tf.keras.layers.GlobalAveragePooling2D()(h2_3)
    h2_3 = tf.reshape(h2_3, [-1, 1, 1, f1])

    multi = tf.multiply(h2_2, h2_3)
    multi = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(multi)
    multi_skip = multi
    concat = layers.Concatenate()([h2_1, multi])

    # Third block
    x = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(concat)
    x = BatchNormalization()(x)


    # add 
    x = x + x_skip + multi_skip
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
    h2_1 = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_1 = BatchNormalization()(h2_1)
    h2_1 = keras.activations.relu(h2_1)

    h2_2 = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_2 = BatchNormalization()(h2_2)
    h2_2 = keras.activations.relu(h2_2)

    h2_3 = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(h1)
    h2_3 = BatchNormalization()(h2_3)
    h2_3 = keras.activations.relu(h2_3)
    h2_3 = tf.keras.layers.GlobalAveragePooling2D()(h2_3)
    h2_3 = tf.reshape(h2_3, [-1, 1, 1, f1])

    multi = tf.multiply(h2_2, h2_3)
    multi = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(multi)
    multi_skip = multi
    concat = layers.Concatenate()([h2_1, multi])

    # Third block
    x = Conv2D(f2, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(concat)
    x = BatchNormalization()(x)

    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=keras.regularizers.l2(0.001), kernel_initializer='he_normal')(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add 
    x = x + x_skip + multi_skip
    x = keras.activations.relu(x)
    x = layers.Dropout(0.3)(x)
    return x

def resnet50_multiply():

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

  # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    
  # ends with average pooling and dense connection

    x = layers.AveragePooling2D((4, 4), padding='same')(x)

    x = layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(11, activation='softmax', kernel_initializer='he_normal')(x)

  # define the model 

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Resnet50_multiply')

    return model


# In[8]:


model_3 = resnet50_multiply()
model_3.summary()


# In[ ]:




