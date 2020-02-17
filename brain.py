import tensorflow as tf
from tensorflow import keras
import numpy as np

#TODO::parsing data for inputs

visible = keras.layers.Input(shape=(100,1))
hidden1 = keras.layers.Dense(10, activation='relu')(visible)
hidden2 = keras.layers.Dense(10, activation='relu')(hidden1)
hidden3 = keras.layers.Dense(10, activation='relu')(hidden2)
output = keras.layers.Dense(1, activation='sigmoid')(hidden3)
model = keras.models.Model(inputs=visible, outputs=output)

#TODO::training commands and saving the weight data

# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file='NNplot.png')
