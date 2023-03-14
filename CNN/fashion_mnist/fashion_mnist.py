# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:01:27 2023

@author: Tolga
"""
#C:\Users\Tolga\Desktop\WorkSpace\spyder_ML\Kod arşivi\DeepLearning\CNN\fashion_mnist

from keras.datasets import fashion_mnist
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#%%
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

print('Training data shape : ', x_train.shape, y_train.shape)

print('Testing data shape : ', x_test.shape, y_test.shape)
#%%
classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
#%%
def plots(index:int):
    vis = x_train.reshape(60000,28,28)
    plt.imshow(vis[index,:,:]) 
    plt.legend()
    plt.axis("off")
    plt.show()
    print(np.argmax(y_train[index]))
#%%
def plots_gray(index:int):
    plt.figure(figsize=[7,7])
    
    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(x_train[index,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(y_train[0]))
    
    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(x_test[index,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(y_test[0]))
    
#%%reshape because that's what keras wants
x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)
print(x_test.shape,x_train.shape)
#%%
# The data right now is in an int8 format, so before you feed it into the network you need to convert its type to float32,
# and you also have to rescale the pixel values in range 0 - 1 inclusive. So let's do that!

x_test=x_test.astype("float32")
x_train=x_train.astype("float32")


#%% 5=[0,0,0,0,0,1,0,0,0,0]
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
print(y_test[6])
#%%
train_X,valid_X,train_label,valid_label = train_test_split(x_train, y_train, test_size=0.2, random_state=13)
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)
#%%
batch_size = 64
epochs = 20
num_classes = 10

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(Dense(num_classes, activation='softmax'))

fashion_model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
#%%
fashion_model.summary()
#%%
from tensorflow.keras.callbacks import TensorBoard
file_name = 'fashion_tensorboar_1'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
   
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,
                                  epochs=epochs,verbose=1,validation_data=(valid_X, valid_label),callbacks=[tensorboard])




#%%
fashion_model.save_weights('fashion_tensorboar_1.h5')  # always save your weights after training or during training

#%% evaluation 
print(fashion_train.history.keys())
plt.plot(fashion_train.history["loss"],label = "Train Loss")
plt.plot(fashion_train.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(fashion_train.history["accuracy"],label = "Train Accuracy")
plt.plot(fashion_train.history["val_accuracy"],label = "Validation Accuracy")
plt.legend()
plt.show()
#%% save history
import json
with open('fashion_tensorboar_1.json', 'w') as f:
    json.dump(fashion_train.history, f)
    
#%% load history
import codecs
with codecs.open("fashion_tensorboar_1.json", 'r', encoding='utf-8') as f:
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

#%%add drop out 
batch_size = 128
epochs = 60
num_classes = 10
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))                  
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))           
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))
fashion_model.summary()

fashion_model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
#%%

file_name = 'fashion_tensorboar_add_dropout_3'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))#python -m tensorboard.main --logdir=logs/
   
fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size=batch_size,
                                  epochs=epochs,verbose=1,validation_data=(valid_X, valid_label),callbacks=[tensorboard])




#%%
fashion_model.save_weights('fashion_tensorboar_add_dropout_3.h5')  # always save your weights after training or during training

#%% evaluation 
print(fashion_train_dropout.history.keys())
plt.plot(fashion_train_dropout.history["loss"],label = "Train Loss")
plt.plot(fashion_train_dropout.history["val_loss"],label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(fashion_train_dropout.history["accuracy"],label = "Train Accuracy")
plt.plot(fashion_train_dropout.history["val_accuracy"],label = "Validation Accuracy")
plt.legend()
plt.show()
#%% save history
import json
with open('fashion_tensorboar_add_dropout_3.json', 'w') as f:
    json.dump(fashion_train_dropout.history, f)
    
#%% load history
import codecs
with codecs.open("fashion_tensorboar_add_dropout.json", 'r', encoding='utf-8') as f:
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

#%%
predicted_classes = fashion_model.predict(x_test)

predicted_classes = np.argmax(np.round(predicted_classes),axis=1)


true_classes = np.argmax(np.round(y_test),axis=1)

print(predicted_classes.shape, true_classes.shape)


#%%


correct = np.where(predicted_classes==true_classes)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], true_classes[correct]))
    plt.tight_layout()
    plt.show()
#%%

incorrect = np.where(predicted_classes!=true_classes)[0]
print ("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], true_classes[incorrect]))
    plt.tight_layout()

#%% Classification Report
from sklearn.metrics import classification_report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(true_classes, predicted_classes, target_names=target_names))
