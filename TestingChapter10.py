#Testing the timeseries_dataset_from_array() function from chapter 10.
import numpy as np
from tensorflow import keras
Numbers=[0,1,2,3,4,5,6,7,8,9]#This represents the training data set.

dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=Numbers[0:-3],#From begginign to the -3th term, not including the -3 term. Not generally good, I usually want it to use all the dataset.
    targets=Numbers[3:],
    sequence_length=3,
    batch_size=2,
)



a,b,c= dummy_dataset
print("a",a)
print("b",b)
print("c",c)
print("Stop")
for inputs, targets in dummy_dataset:#This loops through the batches. Inputs holds the current batch
    print(inputs)
    print(targets)
    for i in range(inputs.shape[0]):#Loops through all the time series. inputs.shape[0] is the number of time series within that batch (most batches have same size, but last batch may have less).
        print(i)
        print([int(x) for x in inputs[i]], int(targets[i]))#It loops through all the values in a specific time series