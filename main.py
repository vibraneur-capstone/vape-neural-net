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
gt = gt[0:104]

def generator(files, groundtruth):
    input1 = np.zeros((1, 4))
    input2 = np.zeros((1, 2176))
    truth = np.zeros((1,1))
    i = 0
    
    for f in files:
        # Open up our file
        input = open(path + f,"r")
        
        # Read our file
        line = np.array(input.readline().split('\t'))
        split = np.split(line, [4])
        input1 = split([0])
        input2 = split([1])
        truth = groundtruth([i])
        
        yield [input1, input2], truth
        i += 1

### Manipulate the model here ###

#m = brain.CreateModel()
#brain.SaveModel(m, 'model.h5')
m = brain.LoadModel('model.h5')

gen = generator(data, gt)
brain.TrainModel(m, './model/model.g5', gen, len(data))