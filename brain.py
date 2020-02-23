# Richard Walmsley
# 17/02/2020
# This source holds functions to make manipulating the model easier.
# It separates dealing with input data parsing and modularizes functions.

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Creates the model. Use this to initialize or reinitialize the model for training.
def CreateModel():
    # Inputs
    inputA = keras.layers.Input(shape=(4,))
    inputB = keras.layers.Input(shape=(10240))

    # First branch
    x = keras.layers.Dense(10, activation="relu")(inputA)

    # Second branch
    y = keras.layers.Dense(1000, activation='relu')(inputB)
    y = keras.layers.Dense(500, activation='relu')(y)
    y = keras.layers.Dense(250, activation='relu')(y)
    y = keras.layers.Dense(100, activation='relu')(y)
    y = keras.layers.Dense(10, activation='relu')(y)

    # Combine these branches
    concatenate = keras.layers.concatenate([x, y])

    # Final layers
    z = keras.layers.Dense(10, activation='relu')(concatenate)
    z = keras.layers.Dense(5, activation='relu')(z)
    z = keras.layers.Dense(1, activation='sigmoid')(z)

    # Final model
    model = keras.models.Model(inputs=[inputA, inputB], outputs = z)

    # Display model architecture
    model.summary()
    
    # Compiles model with predetermined training configuration
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    
    return model

#TODO::Finish code to train a model
# Trains a model to a given a generator, target, epochs and batch size
def TrainModel(model, target, generator, length):
    # Define callbacks to allow training to continue if interrupted
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='./model/%s' % target, monitor='loss', verbose=1, save_best_only=True, mode='min')
    checkpoints = [checkpoint]
    model.fit_generator(generator, steps_per_epoch=1, epochs=length, callbacks=checkpoints)

# Loads in a model given its .h5 file name and creates an instance of it
def LoadModel(target):
    print("Loading model %s..." % target)
    model = keras.models.load_model('./model/%s' % target)
    model.summary()
    model.get_weights()

    return model
    
    # Create a model instance
    #model = create_model()
    
# Saves model as target .h5 file name.
def SaveModel(model, target):
    print("Saving model %s..." % target)
    model.save('./model/%s' % target)