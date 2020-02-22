# Richard Walmsley
# 17/02/2020

import tensorflow as tf
from tensorflow import keras
import numpy as np

#TODO::parsing/set up data for inputs

# Creates the model. Use this to initialize or reinitialize the model for training.
def CreateModel():
    # Inputs
    inputA = keras.layers.Input(shape=(4,))
    inputB = keras.layers.Input(shape=(2140,))

    # First branch
    x = keras.layers.Dense(10, activation="relu")(inputA)

    # Second branch
    y = keras.layers.Dense(500, activation='relu')(inputB)
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
    model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])
    
    return model

# Trains a model to a given set of input and groundtruth data
def TrainModel(model, input, groundtruth):  
    #model.fit(input, groundtruth, epochs=, batch_size=)

    #TODO::Finish code to train a model using checkpoints or from scratch

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