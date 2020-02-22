# Use this file to run commands from brain.py
# From brain.py we can initialize, save, load
# and train the NN as required.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import brain

m = brain.CreateModel()
brain.SaveModel(m, 'model.h5')
m = brain.LoadModel('model.h5')