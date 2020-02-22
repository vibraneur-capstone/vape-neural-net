# Richard Walmsley
# 22/02/2020
# Use this file to run commands from brain
# From brain we can initialize, save, load
# and train the NN as required.

import tensorflow as tf
from tensorflow import keras
from os import listdir
from os.path import isfile, join
import numpy as np
import brain

datapath = './dataset/'
gtpath = './groundtruth/'

#TODO::parsing/set up data for inputs using generator
# Generates a list of files in our datapath
data = [f for f in listdir(datapath) if isfile(join(datapath, f))]

# Generate numpy array of single element arrays for ground truth input
gtfile = open(gtpath + 'gt1.dat', "r")
gt = np.array([[float(i)] for i in gtfile.readlines()])
print(gt)

"""
def BatchGenerator(files):
    for f in files:
        # Open up our file
        input = open(path + file,"r")
        
        # Read our file
        lines = input.readlines()
        
        for x in lines:
            lines.split()
"""

#m = brain.CreateModel()
#brain.SaveModel(m, 'model.h5')
#m = brain.LoadModel('model.h5')