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
from math import ceil
import brain


## Training parameters
modelname = 'model.h5'

dataset = 1

samples = 2156
batches = 4 # Batch size is a multiple of dataset length (2156 = 2*2*7*7*11)
steps = ceil(samples/batches)
epochs = 25
##

# Formats lists of files given a bearing and dataset. Returns path, list of files and groundtruth array
def getData(dataset, bearing):
    dpath = './dataset/output_%i/' % dataset
    gtpath = './groundtruth/'

    # Generates a list of files in path based on dataset and bearing parameters
    data = [f for f in listdir(dpath)
                if isfile(join(dpath, f)) 
                and f.startswith('dataset_%i.bearing_%i.' % (dataset, bearing))] 

    # Generate numpy array of single element arrays for ground truth input
    gtfile = open(gtpath + 'gt%i.dat' % dataset, "r")
    gt = np.array([[float(i)] for i in gtfile.readlines()])
    
    return dpath, data, gt

#TODO:: Add time-step for LSTM layer
#TODO:: Add random data point selection

# Generate batches of a given size, stopping once it reaches the total amount of data
# and streams this to the model iteratively as it trains.
def generator(path, files, groundtruth, batchsize):
    while True:
        # First for loop iterates across the whole dataset length in batch sizes
        for batch in range(0, len(files), batchsize):
        
            # Second for loop iterates through each batch
            # so we yield 2 inputs and ground truth times the batch size
            for i in range(batch, batch + batchsize):
            
                # Get file at specified index
                f = files[i]
            
                # Open up our file
                input = open(path + f,"r")
                
                # Read our file
                line = np.array(input.readline().split(' '))
                
                # Split our input into an array of size (4,) and (10240,)
                # and find corresponding truth value
                split = np.split(line, [4])
                try:
                    input1 = np.array([split[0]]).astype(np.float32)
                    input2 = np.array([split[1]]).astype(np.float32)
                    truth = groundtruth[files.index(f)].astype(np.float32)
                except:
                    print("\nSomething went wrong with your data in file " + f + "\n")
                    continue
                
                # Stream this to the model during training
                yield ([input1, input2], truth)

### Manipulate the model here ###

#m = brain.CreateModel()
#brain.SaveModel(m, modelname)

# Loads in the model and trains it over bearings 1 to 8 in specified dataset
for x in range(1,8):
    print("TRAINING: Dataset %i, Bearing %i" % (dataset, x))

    datapath, datalist, groundtruth = getData(dataset, x) # dataset 1, bearing x

    m = brain.LoadModel(modelname)

    # Create generator instance
    gen = generator(datapath, datalist, groundtruth, batches)

    # Train our model
    brain.TrainModel(m, modelname, gen, steps, epochs)

