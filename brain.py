import tensorflow as tf
from tensorflow import keras
import numpy as np

#TODO::parsing/set up data for inputs

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

# Create a model instance
model = create_model()

#TODO:training
model.compile(optimizer='adam', loss='crossentropy', metrics=['accuracy'])
model.fit(input, groundtruth, epochs=, batch_size=)

#TODO::checkpoint commands to save model, load in previous weights and continue training