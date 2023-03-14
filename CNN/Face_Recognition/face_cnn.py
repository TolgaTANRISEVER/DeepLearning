# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 21:30:25 2023

@author: Tolga
"""


 
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt 
from tensorflow.keras.utils import img_to_array
'''######################## Create CNN deep learning model ########################'''


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense




train_path='Face_Recognition/Final Training Images'
test_path="Face_Recognition/Final Testing Images"

#%%
img= load_img(train_path+"/face2/image_0022_Face_1.jpg")

plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

className = glob(train_path + '/*' )
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)

#%%
className = glob(train_path + '/*' )
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)


a = glob(train_path+"/face2"+"/*")
print(len(a)*len(className))


#%% Data generation Train test
# Data preprocessing
batch_size = 32# her iteresyonda 32 resim train ediyor


train_datagen = ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
 
# Defining pre-processing transformations on raw images of testing data
# No transformations are done on the testing images
test_datagen = ImageDataGenerator()
 
# Generating the Training Data
train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')
 
 
# Generating the Testing Data
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

# Printing class labels for each face
test_generator.class_indices
#%%





 
# CNN model Create
model=Sequential()

#1.CNN Layer
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))#feature map ,feature sayısı
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
              optimizer = "adam",
              metrics = ["accuracy"])
#%%
from tensorflow.keras.callbacks import TensorBoard
file_name = 'face_hist_1'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))
train_steps = train_generator.samples // train_generator.batch_size
hist = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps,
        epochs=50,
        validation_data=test_generator,
        validation_steps=10,
        callbacks=[tensorboard])

#%% model save
model.save_weights("fave_his_1.h5")

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
#%%

'''############ Creating lookup table for all faces ############'''
# class_indices have the numeric tag for each face
TrainClasses=train_generator.class_indices
 
# Storing the face and the numeric tag for future reference
ResultMap={}
for faceValue,faceName in zip(TrainClasses.values(),TrainClasses.keys()):
    ResultMap[faceValue]=faceName
 
# Saving the face map for future reference
import pickle
with open("ResultsMap.pkl", 'wb') as fileWriteStream:
    pickle.dump(ResultMap, fileWriteStream)
 
# The model will give answer as a numeric tag
# This mapping will help to get the corresponding face name for it
print("Mapping of Face and its ID",ResultMap)
 
# The number of neurons for the output layer is equal to the number of faces
OutputNeurons=len(ResultMap)
print('\n The Number of output neurons: ', OutputNeurons)
'''########### Making single predictions ###########'''
import numpy as np

 #%%
ImagePath='C:/Users/Tolga/Desktop/WorkSpace/spyder_ML/Kod arşivi/DeepLearning/CNN/Face_Recognition/Face_Recognition/Final Testing Images/face15/1face15.jpg'
test_image=load_img(ImagePath,target_size=(64, 64))
test_image=img_to_array(test_image)
 
test_image=np.expand_dims(test_image,axis=0)
 
result=model.predict(test_image,verbose=0)
#print(training_set.class_indices)
 
print('####'*10)
print('Prediction is: ',ResultMap[np.argmax(result)])



























