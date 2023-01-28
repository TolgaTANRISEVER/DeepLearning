# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 14:49:31 2023

@author: Tolga
"""

"""
Tolga Tanrısever And ChatGpt logistic reg vs 

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
def dummies(train_df:pd.DataFrame,columns):
    from sklearn import preprocessing
    le=preprocessing.LabelEncoder()
    train_df[columns]=le.fit_transform(train_df[columns])
    
    print(train_df)
    
    train_df=pd.get_dummies(train_df,columns=[columns])
    return train_df
#%%normalization
def normalzier(train_df:pd.DataFrame):
    from sklearn.preprocessing import Normalizer
    nr=Normalizer() 
    train_df=pd.DataFrame(nr.fit_transform(train_df))
    return train_df
def scaler(train_df:pd.DataFrame):
    from sklearn.preprocessing import StandardScaler 
    sc=StandardScaler()
    train_df=sc.fit_transform(train_df)
    return train_df
#%% 1. yöntem başarı sırası scaler>2.yöntem>normalizer

data=pd.read_csv("data.csv",sep=",")
data.drop(labels=["id","Unnamed: 32"],axis=1,inplace=True)
pd.DataFrame(dummies(data, "diagnosis"))
y=data.diagnosis.values
x=data.drop(["diagnosis"],axis=1)


x_copy=x.copy()
x=scaler(x)



# a=list(x_copy.columns)
# x.columns=a
#%% 2.yöntem
# # read csv
# data = pd.read_csv("data.csv")
# data.drop(["Unnamed: 32","id"],axis=1,inplace = True)
# data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
# print(data.info())

# y = data.diagnosis.values
# x_data = data.drop(["diagnosis"],axis=1)

# # normalization
# x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# # (x - min(x))/(max(x)-min(x))


#%%  train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

#%%
# lets initialize parameters
# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b

#%%Forward Propagation

# calculation of z
#z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# Forward propagation steps:
# find z = w.T*x+b
# y_head = sigmoid(z)
# loss(error) = loss(y,y_head)
# cost = sum(loss)
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z) # probabilistic 0-1
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    return cost 

#%%forward_backward_propagation
# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients

#%%update
# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            #print ("Cost after iteration %i: %f" %(i, cost))
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)

 #%% prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    y_head = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(y_head.shape[1]):
        if y_head[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test)


#%%logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return y_prediction_test , y_prediction_train,cost_list
#%%
y_prediction_test , y_prediction_train,cost_list=logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 2500)












