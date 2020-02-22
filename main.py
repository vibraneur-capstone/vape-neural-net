# Richard Walmsley
# 22/02/2020
# Use this file to run commands from brain
# From brain we can initialize, save, load
# and train the NN as required.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import brain

#TODO::parsing/set up data for inputs

m = brain.CreateModel()
brain.SaveModel(m, 'model.h5')
m = brain.LoadModel('model.h5')