# Richard Walmsley
# 17/02/2020
# This source holds functions to make manipulating the model easier.
# It separates dealing with input data parsing and modularizes functions.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Creates the model. Use this to initialize or reinitialize the model for training.
def CreateModel():
    # Inputs
    inputA = keras.layers.Input(shape=(4,))
    inputB = keras.layers.Input(shape=(10240,))

    # First branch
    x = keras.layers.Dense(10, activation='relu')(inputA)
    x = keras.layers.LeakyReLU(alpha=0.05)(x)

    #TODO:: Get LSTM layer to work

    # Second branch
    #y = keras.layers.LSTM(500, input_shape=(10240,1), activation='relu', return_sequences=False)(inputB)
    y = keras.layers.Dense(250, activation='relu')(inputB)
    y = keras.layers.LeakyReLU(alpha=0.05)(y)
    #y = keras.layers.Dropout(rate=0.1)(y) # Dropout layer with 20% dropout to prevent overfitting
    y = keras.layers.Dense(250, activation='relu')(y)
    y = keras.layers.LeakyReLU(alpha=0.05)(y)
    y = keras.layers.Dense(10, activation='relu')(y)

    # Combine these branches
    concatenate = keras.layers.concatenate([x, y])

    # Final layers
    z = keras.layers.Dense(20, activation='relu')(concatenate)
    z = keras.layers.LeakyReLU(alpha=0.05)(z)
    z = keras.layers.Dense(1)(z)

    # Final model
    model = keras.models.Model(inputs=[inputA, inputB], outputs = z)

    # Display model architecture
    model.summary()
    
    # Generate optimizer
    adam = keras.optimizers.Adam(learning_rate=0.000000000001)
    
    # Compiles model with predetermined training configuration
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    
    return model

# Trains a model to a given a generator, target, epochs and step size (batches done by generator)
def TrainModel(model, target, generator, number_of_steps, number_of_epochs):
    # Define callbacks to allow training to continue if interrupted
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./model/%s' % target, monitor='loss', verbose=1, save_best_only=True, mode='min')
    model.fit_generator(generator, steps_per_epoch=number_of_steps, epochs=number_of_epochs, verbose=1, callbacks=[checkpointer])

def EvaluateModel(model, generator, number_of_steps):
    model.evaluate_generator(generator, steps=number_of_steps, verbose=1)
    
def PredictModel(model, generator, number_of_steps):
    model.predict_generator(generator, steps=number_of_steps, verbose=1)

'''  
def PlotModel(history, validation):
    # Plot training and validation loss values
    plt.plot(history.history['loss'])
    plt.plot(validation.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training and validation mean absolute error values
    plt.plot(history.history['mae'])
    plt.plot(validation.history['val_mae'])
    plt.title('Model Error')
    plt.ylabel('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
'''

# Loads in a model given its .h5 file name and creates an instance of it
def LoadModel(target):
    print("Loading model %s..." % target)
    model = keras.models.load_model('./model/%s' % target)
    #model.summary()
    #model.get_weights()

    return model
    
# Saves model as target .h5 file name.
def SaveModel(model, target):
    print("Saving model %s..." % target)
    model.save('./model/%s' % target)