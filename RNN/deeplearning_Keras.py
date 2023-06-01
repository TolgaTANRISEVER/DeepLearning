# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:00:08 2023

@author: Tolga
"""


#Recurrent Neural Network (RNN) with Keras

import pandas as  pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
#%%
train_df=pd.read_csv("C:/Users/Tolga/Desktop/WorkSpace/spyder_ML/Kod arşivi/DeepLearning/RNN/Recurrent_Neural_Network/Stock_Price_Train.csv")

train_df_open=train_df.Open.values
#%%


mmx=MinMaxScaler(feature_range=(0, 1))
minmax=mmx.fit_transform(train_df_open.reshape(-1,1))
 




#%%
fig=px.line(minmax,labels={'x':'t', 'y':'sin(t)'})
fig.show("browser")
#%%
# Creating a data structure with 50 timesteps and 1 output
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 1258):
    X_train.append(minmax[i-timesteps:i, 0])
    print(i,timesteps)
    y_train.append(minmax[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#%%
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_train
#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from tensorflow import keras
import tensorflow as tf



# Initialising the RNN
regressor = Sequential()

# Adding the first RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50,activation='tanh', return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth RNN layer and some Dropout regularisation
regressor.add(SimpleRNN(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
#tensorboard
from tensorflow.keras.callbacks import TensorBoard
file_name = 'my_saved_model_3'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 5, batch_size = 10,callbacks=[tensorboard])

  
#%%
# Getting the real stock price of 2017
dataset_test = pd.read_csv('C:/Users/Tolga/Desktop/WorkSpace/spyder_ML/Kod arşivi/DeepLearning/RNN/Recurrent_Neural_Network/Stock_Price_Test.csv')

#%%
real_stock_price = dataset_test.loc[:, ["Open"]].values
#%%


# Getting the predicted stock price of 2017
dataset_total = pd.concat((train_df['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timesteps:].values.reshape(-1,1)
inputs = mmx.transform(inputs)  # min max scaler

#%%



X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = mmx.inverse_transform(predicted_stock_price)
#%%
# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
# epoch = 250 daha güzel sonuç veriyor.


#%%LSTM
from keras.layers import LSTM
data = pd.read_csv('C:/Users/Tolga/Desktop/WorkSpace/spyder_ML/Kod arşivi/DeepLearning/RNN/Recurrent_Neural_Network/international-airline-passengers.csv',skipfooter=5)

dataset = data.iloc[:,1].values
plt.plot(dataset)
plt.xlabel("time")
plt.ylabel("Number of Passenger")
plt.title("international airline passenger")
plt.show()

#%%
dataset = dataset.reshape(-1,1)#kerans want this shape
dataset = dataset.astype("float32")
dataset.shape

#%% scaling 
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#%%
train_size = int(len(dataset) * 0.50)
test_size = len(dataset) - train_size
train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))

#%%

time_stemp = 10
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = np.array(dataX)
trainY = np.array(dataY)  

#%%


dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = np.array(dataX)
testY = np.array(dataY)  

#%%

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
#%%
# model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
model = Sequential()
model.add(LSTM(10,activation="tanh", input_shape=(1, time_stemp),
               recurrent_activation="sigmoid",recurrent_dropout = 0,
               unroll=False,use_bias=True)) # 10 lstm neuron(block)
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
from tensorflow.keras.callbacks import TensorBoard
file_name = 'my_saved_model_3'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
model.fit(trainX, trainY, epochs=200, batch_size=1,callbacks=[tensorboard])





#%%
from sklearn.metrics import mean_squared_error
import math
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


#%%
# shifting train
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan

trainPredictPlot[time_stemp:len(trainPredict)+time_stemp, :] = trainPredict
# shifting test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stemp*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()