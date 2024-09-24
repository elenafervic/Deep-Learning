#Testing the timeseries_dataset_from_array() function from chapter 10.
import numpy as np
from tensorflow import keras
Numbers=[0,1,2,3,4,5,6,7,8,9]#This represents the training data set.

dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=Numbers[0:-3],#From begginign to the -3th term, not including the -3 term. Not generally good, I usually want it to use all the dataset.
    targets=Numbers[3:],
    sequence_length=3,
)

#Dummy dataset is a group of tuples, one tuple per batch eg if we had 2 batches: (TimeSeries_Batch_1,Targets_Batch_1),(TimeSeries_Batch_2,Targets_Batch_2)
#Only way I can figure out to separate a tuple is:
for TimeSeries_Batch,Targets_Batch in dummy_dataset:
    #Use within the loop, or afterwards too if there is only one batch.

#a=dummy_dataset[0] This doesn't work.

#If there were three different batches you could do this to separate all three tuples of batches from each other.
a,b,c= dummy_dataset
print("a",a)
print("b",b)
print("c",c)