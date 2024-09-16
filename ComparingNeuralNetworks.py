# We are testing things to help me understand this kaggle article better https://www.kaggle.com/code/asortubay/pytorch-comparing-dm-conv-lstm-gru#LETS-START-PLAYING-WITH-DEEP-LEARNING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_sunspots = pd.read_csv("/Users/elena/Coding/Deep Learning/Sunspots.csv")
#print(df_sunspots.shape)
#print(df_sunspots.head())


#Preparing the data by splitting it into training, validation and testing.
total_rows = len(df_sunspots)
train_size = int(total_rows * 0.7)
val_size = int(total_rows * 0.15)

train_data = df_sunspots.iloc[:train_size]#Basically iloc makes separate dataframes when we input [:train_size].
val_data = df_sunspots.iloc[train_size:train_size+val_size]
test_data = df_sunspots.iloc[train_size+val_size:]

#---------------------------------------------------------------------------------------
val_predicted_prev_month = val_data.shift(1)['Monthly Mean Total Sunspot Number'] 
print(val_predicted_prev_month.head())
val_predicted_prev_month = val_data.shift(1)['Monthly Mean Total Sunspot Number'].values
test_predicted_prev_month = test_data.shift(1)['Monthly Mean Total Sunspot Number'].values
val_predicted_prev_month = val_predicted_prev_month[1:] #removing the first month
test_predicted_prev_month = test_predicted_prev_month[1:] #removing the first month

#This actually seems to do the same thing.
val_predicted_prev_month = val_data.shift(1)['Monthly Mean Total Sunspot Number'].values[1:] #first month cannot be predicted so remove it
test_predicted_prev_month = test_data.shift(1)['Monthly Mean Total Sunspot Number'].values[1:]
print(val_data['Monthly Mean Total Sunspot Number'].values[1:])
#computing MSE:
val_mse_prev_month = np.mean((val_predicted_prev_month - val_data['Monthly Mean Total Sunspot Number'].values[1:])**2)#Find the difference between months squared, and then find the mean over all the months.
test_mse_prev_month = np.mean((test_predicted_prev_month - test_data['Monthly Mean Total Sunspot Number'].values[1:])**2)

print("Baseline 1 validation MSE:", val_mse_prev_month)
print("Baseline 1 testing MSE:", test_mse_prev_month)

#MSE is Mean Square Error

