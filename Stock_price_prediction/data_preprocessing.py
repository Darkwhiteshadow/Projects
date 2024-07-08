import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense 
import matplotlib.pyplot as plt

df = pd.read_csv('MSFT.csv')

df1 = df.reset_index()['close']

plt.figure(figsize=(10,6))
plt.plot(df1,color='blue',label='Close price')
plt.title('Microsoft (MSFT) Stock Close Prices')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# preprocessing data 

scaler = MinMaxScaler(feature_range=(0,1))

df1_scaled = scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size = int(len(df1_scaled)*0.7)
test_size = len(df1_scaled)- training_size

train_data,test_data = df1_scaled[0:training_size,:],df[training_size:,:1]

#function to create dataset with sliding window
def create_dataset(dataset,time_step=1):
    dataX,dataY = [],[]
    