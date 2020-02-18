import tensorflow as tf
from tensorflow import keras
import numpy as np

#TODO::parsing data for inputs

# Inputs
inputA = keras.layers.Input(shape=(4,))
inputB = keras.layers.Input(shape=(2140,))

# First branch
x = keras.layers.Dense(10, activation="relu")(inputA)
#x = keras.models.Model(inputs = inputA, outputs = x)

# Second branch
y = keras.layers.Dense(500, activation='relu')(inputB)
y = keras.layers.Dense(250, activation='relu')(y)
y = keras.layers.Dense(100, activation='relu')(y)
y = keras.layers.Dense(10, activation='relu')(y)
#y = keras.models.Model(inputs = inputB, outputs = y)

# Combine these branches
concatenate = keras.layers.concatenate([x, y])

# Final layers
z = keras.layers.Dense(10, activation='relu')(concatenate)
z = keras.layers.Dense(5, activation='relu')(z)
z = keras.layers.Dense(1, activation='sigmoid')(z)

# Final model
model = keras.models.Model(inputs=[inputA, inputB], outputs = z)

# summarize layers
print(model.summary())

#TODO::training commands and saving the weighted data