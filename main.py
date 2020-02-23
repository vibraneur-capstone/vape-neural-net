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
modelname = 'model.h5'

# Generates a list of files in our datapath
data = [f for f in listdir(datapath) if isfile(join(datapath, f))]

# Generate numpy array of single element arrays for ground truth input
gtfile = open(gtpath + 'gt1.dat', "r")
gt = np.array([[float(i)] for i in gtfile.readlines()])
gt = np.split(gt,[104])[0]

def generator(path, files, groundtruth):
    i = 0
    
    for f in files:
        # Open up our file
        input = open(path + f,"r")
        
        # Read our file
        line = np.array(input.readline().split(' '))
        
        # Split our input into arrays of size (4,) and (10240,), and assign truth value
        split = np.split(line, [4])
        input1 = np.array([split[0]]).astype(np.float)
        input2 = np.array([split[1]]).astype(np.float)
        truth = groundtruth[i].astype(np.float)
        
        # Give this to the model
        yield [input1, input2], truth
        i += 1

### Manipulate the model here ###

m = brain.CreateModel()
#brain.SaveModel(m, modelname)
#m = brain.LoadModel(modelname)

# Create generator instance
gen = generator(datapath, data, gt)

# Train our model
brain.TrainModel(m, modelname, gen, len(data))