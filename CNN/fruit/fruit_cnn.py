# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 11:14:26 2023

@author: Tolga
"""
#C:\Users\Tolga\Desktop\WorkSpace\spyder_ML\Kod arşivi\DeepLearning\CNN\fruit
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt
from glob import glob
#%%
train_path="fruits-360_dataset/fruits-360/Training"
test_path="fruits-360_dataset/fruits-360/Test"
#%%
img= load_img(train_path+"/Apple Braeburn/1_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

className = glob(train_path + '/*' )
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)



#%%CNN

# CNN model Create
model=Sequential()

#1.CNN Layer
model.add(Conv2D(32, (3,3),input_shape=x.shape))#feature map ,feature sayısı
model.add(Activation("relu"))# if x=" - " :f(x)=0 elif x=" + " : f(x)=x
model.add(MaxPooling2D())

#3.CNN Layer
model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())


#3.CNN Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

# Fully Connected
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(numberOfClass)) # output sayısı class kadar olur
model.add(Activation("softmax"))


# Model compilation
model.compile(loss = "categorical_crossentropy", # because my data is multiclass
              optimizer = "rmsprop",
              metrics = ["accuracy"])



# Data preprocessing
batch_size = 32# her iteresyonda 32 resim train ediyor


#%%
a = glob(train_path+"/Apple Braeburn"+"/*")
print(len(a)*len(className))
#bu yeterli resim değil cnn için o yüzden image generator kullanıcaz
#this is not enough image for cnn so we will use image generator


#%% Data generation Train test
train_datagen=ImageDataGenerator(rescale=1./255,
                   shear_range=0.3,#change random axis over 360
                   horizontal_flip=True,#random rgiht or left change 
                   zoom_range=0.3
                   )

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_path,
    target_size=(100,100),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")


test_generator=test_datagen.flow_from_directory(
    test_path,
    target_size=(100,100),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")
#%%
from tensorflow.keras.callbacks import TensorBoard
file_name = 'cnn_fruit_hist_3'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 1600 // batch_size,
        epochs=300,
        validation_data = test_generator,
        validation_steps = 800 // batch_size,
        callbacks=[tensorboard])

#%% model save
model.save_weights("cnn_fruit_hist_3.h5")

#%% model evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()

#%% save history
import json
with open("cnn_fruit_hist_3.json","w") as f:
    json.dump(hist.history, f)
    
#%% load history
import codecs
with codecs.open("cnn_fruit_hist_3.json", "r",encoding = "utf-8") as f:
    h = json.loads(f.read())
plt.plot(h["loss"], label = "Train Loss")
plt.plot(h["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"], label = "Train acc")
plt.plot(h["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()   












