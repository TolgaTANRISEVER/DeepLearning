# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:57:43 2023

@author: Tolga
"""
#C:\Users\Tolga\Desktop\WorkSpace\spyder_ML\Kod arşivi\DeepLearning\CNN\mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Wrapper, BatchNormalization
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
warnings.filterwarnings("ignore")
#%% 
# load and preprocess
test=pd.read_csv("mnist_data_set/mnist_test.csv")
train=pd.read_csv("mnist_data_set/mnist_train.csv")
def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.values
    np.random.shuffle(data)
    x = data[:,1:].reshape(-1,28,28,1)/255.0#reshape and normalize
    y = data[:,0].astype(np.int32)
    y = to_categorical(y, num_classes=len(set(y)))

    return x,y

train_data_path = "mnist_data_set/mnist_test.csv"
test_data_path = "mnist_data_set/mnist_train.csv"

x_train,y_train = load_and_preprocess(train_data_path)
x_test, y_test = load_and_preprocess(test_data_path)
#%% visualize
X_t = train.drop(labels = ["label"],axis = 1) 
xtr=X_t.copy()
def C(number_of_value:int):
    img = xtr.iloc[number_of_value].values
    img = img.reshape((28,28))
    title=str(train.iloc[number_of_value,0])#labels
    fig = px.imshow(img, color_continuous_scale="gray",title=title)
    fig.show()

#%% CNN
numberOfClass = y_train.shape[1]

model = Sequential()

model.add(Conv2D(input_shape = (28,28,1), filters = 16, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 64, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 128, kernel_size = (3,3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units = 256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(units = numberOfClass))
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
#%%fit
from tensorflow.keras.callbacks import TensorBoard
file_name = 'mnist_4'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs= 25, batch_size= 4000,callbacks=[tensorboard])


#%%
model.save_weights('cnn_mnist_model.h5')  # always save your weights after training or during training
#%% evaluation 
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "Train Loss")
plt.plot(hist.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"],label = "Train Accuracy")
plt.plot(hist.history["val_accuracy"],label = "Validation Accuracy")
plt.legend()
plt.show()

#%% save history
import json
with open('cnn_mnist_hist_1.json', 'w') as f:
    json.dump(hist.history, f)
    
#%% load history
import codecs
with codecs.open("cnn_mnist_hist_1.json", 'r', encoding='utf-8') as f:
    h = json.loads(f.read())

plt.figure()
plt.plot(h["loss"],label = "Train Loss")
plt.plot(h["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"],label = "Train Accuracy")
plt.plot(h["val_accuracy"],label = "Validation Accuracy")
plt.legend()
plt.show()




