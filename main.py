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
import matplotlib.pyplot as plt

## Options and Settings
modelname = 'model.h5'

create = True
load = False
train = True
evaluate = True
predict = True
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
    
    ## Training parameters
    samples = len(data)
    batches = 4 # Safe batch size
    steps = ceil(samples/batches)
    epochs = 30
    ##
    
    return dpath, data, gt, samples, batches, steps, epochs

#TODO:: Add time-step for LSTM layer
#TODO:: Add random data point selection

# Generate batches of a given size, stopping once it reaches the total amount of data
# and streams this to the model iteratively as it trains.
def train_generator(path, files, groundtruth, batchsize):
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
                    #input2 = np.array([split[1]]).astype(np.float32)
                    truth = groundtruth[files.index(f)].astype(np.float32)
                except:
                    print("\nSomething went wrong with your data in file " + f + "\n")
                    continue
                
                # Stream this to the model during training
                #yield ([input1, input2], truth)
                yield(input1, truth)

def eval_generator(path, files, batchsize):
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
                    #input2 = np.array([split[1]]).astype(np.float32)
                except:
                    print("\nSomething went wrong with your data in file " + f + "\n")
                    continue
                
                # Stream this to the model during training
                #yield ([input1, input2])
                yield(input1)

### Manipulate the model here ###

if create:
    m = brain.CreateModel()
    brain.SaveModel(m, modelname)

if load:
    m = brain.LoadModel(modelname)

# Training for dataset 1 and 3
if train:

    dataset=1
    for y in range(1,9):
        print("TRAINING: Dataset %i, Bearing %i" % (dataset, y))

        datapath, datalist, groundtruth, samples, batches, steps, epochs = getData(dataset, y) # dataset x, bearing y

        # Create generator instance
        gen = train_generator(datapath, datalist, groundtruth, batches)

        # Train our model
        history = brain.TrainModel(m, modelname, gen, steps, epochs)
    
    dataset = 3
    for y in range(1,5):
        print("TRAINING: Dataset %i, Bearing %i" % (dataset, y))

        datapath, datalist, groundtruth, samples, batches, steps, epochs = getData(dataset, y) # dataset x, bearing y

        # Create generator instance
        gen = train_generator(datapath, datalist, groundtruth, batches)

        # Train our model
        history = brain.TrainModel(m, modelname, gen, steps, epochs)

# Evaluation for dataset 2
if evaluate:
    for z in range(1,5):
        print("EVALUATING: Dataset %i, Bearing %i" % (2, z))
        
        datapath, datalist, groundtruth, samples, batches, steps, epochs = getData(2, z) # dataset 2, bearing y

        # Create generator instance
        gen = train_generator(datapath, datalist, groundtruth, batches)

        # Evaluate the model
        results = brain.EvaluateModel(m, gen, steps)

        print("Evaluation metrics: ", results)

# Prediction for dataset 2
if predict:
    for p in range(1,5):
        print("PREDICTING: Dataset %i, Bearing %i" % (2, p))
        
        datapath, datalist, groundtruth, samples, batches, steps, epochs = getData(2, p) # dataset 2, bearing p

        # Create generator instance
        gen = eval_generator(datapath, datalist, batches)

        # Evaluate the model
        prediction = brain.PredictModel(m, gen, steps)

        print("Prediction: ", prediction)

# Testing out predictions because predict_generator isn't working
datapath, datalist, groundtruth, samples, batches, steps, epochs = getData(2, 4)

for f in datalist:
    # Open up our file
    input = open(datapath + f,"r")

    # Read our file
    line = np.array(input.readline().split(' '))

    # Split our input into an array of size (4,) and (10240,)
    # and find corresponding truth value
    split = np.split(line, [4])
    input1 = np.array([split[0]]).astype(np.float32)
    #input2 = np.array([split[1]]).astype(np.float32)
    truth = groundtruth[datalist.index(f)].astype(np.float32)

    print("Truth: ", truth, ", Prediction: ", m.predict(input1))#m.predict([input1, input2])))

#brain.PlotModel(history, results)

brain.SaveModel(m, modelname)