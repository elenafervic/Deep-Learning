#Testing the timeseries_dataset_from_array() function from chapter 10.
import numpy as np
from tensorflow import keras
Numbers=[0,1,2,3,4,5,6,7,8,9]#This represents the training data set.

dummy_dataset = keras.utils.timeseries_dataset_from_array(
    data=Numbers[0:-3],#From begginign to the -3th term, not including the -3 term.
    targets=Numbers[3:],
    sequence_length=3,
    batch_size=2,
)
print(dummy_dataset)


for inputs, targets in dummy_dataset: 
    print(inputs, targets)
    for i in range(inputs.shape[0]):
        print([int(x) for x in inputs[i]], int(targets[i]))