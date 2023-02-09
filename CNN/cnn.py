# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:13:44 2023
Convolutional Neural Networks (CNN)
@author: Tolga
"""
#C:\Users\Tolga\Desktop\WorkSpace\spyder_ML\Kod arşivi\DeepLearning
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import plotly.express as px
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from sklearn.model_selection import train_test_split
#%%
train_df=pd.read_csv("cnn_data_set/trainCNN.csv")
test_df=pd.read_csv("cnn_data_set/testCNN.csv")

# put labels into y_train variable
Y_train = train_df["label"]
# Drop 'label' column
X_train = train_df.drop(labels = ["label"],axis = 1) 

#%%
def hisplot(variable):
    fig = px.histogram(Y_train,
                       x=variable,
                       color=variable,
                       title=variable,
                       width=700,
                       height=500,
                       barmode='overlay'
                       )
    fig.update_layout(bargap=0.1,barmode='stack')
    fig.update_xaxes(categoryorder="category ascending")
   
    fig.show("browser")
    
# hisplot("label")
#%%
# plot some samples
xtr=X_train.copy()
def C(number_of_value:int):
    img = xtr.iloc[number_of_value].values
    img = img.reshape((28,28))
    title=str(train_df.iloc[number_of_value,0])#labels
    fig = px.imshow(img, color_continuous_scale="gray",title=title)
    fig.show()


fig = px.colors.sequential.swatches_continuous()
fig.show()
#%% normalization
"""
Normalization, Reshape and Label Encoding
Normalization
We perform a grayscale normalization to reduce the effect of illumination's differences.
If we perform normalization, CNN works faster.
Reshape
Train and test images (28 x 28)
We reshape all data to 28x28x1 3D matrices.
Keras needs an extra dimension in the end which correspond to channels. Our images are gray scaled so it use only one channel.
Label Encoding
Encode labels to one hot vectors
2 => [0,0,1,0,0,0,0,0,0,0]
4 => [0,0,0,0,1,0,0,0,0,0]
"""
# Normalize the data
print("x_train shape: ",X_train.shape)
print("test shape: ",test_df.shape)
X_train = X_train / 255.0
test_df = test_df / 255.0

# Reshape
print("x_train shape: ",X_train.shape)
print("test shape: ",test_df.shape)
X_train = X_train.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)

# Label Encoding 

Y_train = to_categorical(Y_train, num_classes = 10)

#another way
# def dummies(train_df,columns):
#     from sklearn import preprocessing
#     le=preprocessing.LabelEncoder()
#     train_df[columns]=le.fit_transform(train_df[columns])
    
#     print(train_df)
    
#     train_df=pd.get_dummies(train_df,columns=[columns])
#     return train_df
# Y_train=dummies(pd.DataFrame(Y_train),"label")
## pd to series

# Y_train..squeeze()



## convert dataframe to numpy array of float

# Y_train = df.Y_train.astype(np.float)


print("x_train shape: ",X_train.shape)
print("test shape: ",test_df.shape)
print("x_train shape: ",X_train.shape)
print("test shape: ",test_df.shape)
#%%train _test split 
X_train ,X_val ,Y_train ,Y_val=train_test_split(X_train, Y_train, test_size=0.3, random_state=42)

#%%Create model





# conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
# Dropout: Dropout is a technique where randomly selected neurons are ignored during training
from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential #create model layer barındıran yapı  
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

import keras
##-------------------------
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
# sess = tf.compat.v1.Session(config=config) 
# tf.compat.v1.keras.backend.set_session(sess)
##-------------------------
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
##-------------------------

model = keras.Sequential()
#convulation layer
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', # same padding etrafına 0 matiris ekliyor resimin
                 activation ='relu', input_shape = (28,28,1)))#sadece burda yapıyoruz sonrasında zaten cnn biliyor input_shape
model.add(MaxPool2D(pool_size=(2,2)))#2,2 ye lik bir alandaki max değeri alıyor
model.add(Dropout(0.25))#nöronlar arrasında ki bağlantıyı devre dışı bırakıyor overfitting engeliyor
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))#nöronlar arrasında ki bağlantıyı devre dışı bırakıyor overfitting engeliyor
# fully connected
model.add(Flatten())

model.add(Dense(256, activation = "relu"))#hidden alyer
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))# output layer for multy class 

#%% Read for model.add(Flatten())
#please read this section
"""
  matrix flatten 1 2 
                #3 4  = 1
                        2
                        3
                        4 
Convolutional Neural Networks (CNNs) genellikle resim veya görüntü verileri ile çalışır ve 
bu verilerin işlenmesi için üç boyutlu tensörler kullanılır. Ancak, bu tensörlerin içeriği 
sınıflandırma veya regresyon modelleri için verilen ağlar tarafından işlenmesi gerekir. 
Bu nedenle, üç boyutlu tensörler tek boyutlu dizilere dönüştürülmelidir. Bu işlem "flatten" 
olarak adlandırılır ve bu tensörlerin tüm elemanlarını tek boyutlu bir dizi haline getirir.

Ayrıca, çoklu tabaka katmanlarının bulunduğu CNN ağı modellerinde, her tabaka katmanı 
sadece tek boyutlu girdileri kabul eder. Bu nedenle, tensörler tek boyutlu dizilere 
dönüştürülmeden önceki katmanlardaki çıktıları işlemek mümkün değildir.

Özet olarak, "flatten" işlemi, üç boyutlu tensör verilerinin sınıflandırma veya regresyon
modelleri tarafından işlenmesini veya çoklu tabaka katmanlarının bulunduğu ağ modellerinde
tensör verilerinin işlenmesini sağlar.

English explanation:

In Convolutional Neural Networks (CNNs), images or visual data is often processed,
and three-dimensional tensors are used to represent this data. However, 
the contents of these tensors need to be processed by classification or 
regression models in the network. Therefore, these three-dimensional tensors
must be transformed into one-dimensional arrays. This process is called "flattening" 
and it converts the entire contents of the tensors into a one-dimensional array.

Additionally, in CNN models with multiple layer networks, each layer accepts
only one-dimensional inputs. Therefore, the outputs from previous layers cannot
be processed without transforming the tensors into one-dimensional arrays.

In summary, the "flattening" process enables three-dimensional tensor data 
to be processed by classification or regression models, or processed in a network
model with multiple layer networks.
 ------
DROPOUT

Dropout, Convolutional Neural Networks (CNNs) için bir düzenleme tekniğidir. 
Bu teknik, ağın öğrenmesi sırasında overfitting (aşırı uyum) olasılığını azaltmak
için kullanılır. Overfitting, veriler üzerinde performansın artmasına rağmen, 
veriler dışındaki verilere genellemekte zayıflama olarak tanımlanır. Dropout, 
ağdaki bazı ağırlıkların rastgele seçilen yollarla açık kalması veya atılmasını içerir.
Bu, ağın her bir örnek için ağırlıkların farklı bir şekilde uyarlanmasını ve daha az 
bağımlı hale gelmesini sağlar.

Özet olarak, Dropout, Convolutional Neural Networks (CNNs) ağlarının performansını iyileştirmek
ve overfitting olasılığını azaltmak için kullanılan bir düzenleme tekniğidir.
 
English explanation:

Dropout is a regularization technique used in Convolutional Neural Networks (CNNs)
to reduce the likelihood of overfitting. Overfitting refers to the scenario where
a model performs well on the training data but fails to generalize to new data. 
Dropout involves randomly dropping some weights in the network during the training process.
This forces the network to learn with different weights each time and results in a less
dependent and more robust model.

In summary, Dropout is a regularization technique used to improve the performance of
Convolutional Neural Networks (CNNs) and reduce the risk of overfitting.
"""

#%%
# Define the optimizer modele öğretme 
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

#%%   #please read this section
"""
 Adam optimizer'ın bir örneği oluşturulur ve "optimizer" değişkenine atanır. Adam optimizer,
derin öğrenmede popüler bir optimize edicidir ve gradient descent ve Momentum optimizasyon
tekniklerinin avantajlarını birleştirir.

Bu optimizer, 3 adet parametre alır:

 lr (learning rate): modelin ağırlıklarını güncelleme sıklığını belirler. Yüksek bir 
learning rate hızlı öğrenme ancak overshooting (hedefi kaçırmak) riskini arttırır. 
Düşük bir learning rate ise yavaş öğrenme ancak overshooting riskini azaltır.

 beta_1: bu parametre, Momentum optimizasyon tekniğinin momentum değerini belirler.

 beta_2: bu parametre, optimizasyon için kullanılan RMSprop optimizasyon tekniğinin bir beta değeridir.

Bu optimizer, modelin ağırlıklarını etkin bir şekilde optimize etmek için Momentum ve RMSprop 
tekniklerinin avantajlarını birleştirir ve hızlı ve doğru bir converge sağlar.

 Momentum: Momentum, gradient descent optimize etme sürecinin hızlandırılması 
için kullanılan bir optimize edici tekniktir. Momentum'un temel fikri, bir
nesnenin belirli bir yönde hareket etmesi halinde, gradient değişse bile 
o yönde hareket etme eğiliminde olmasıdır. Derin öğrenmede, Momentum optimize
etme tekniği, geçmiş iterasyonların birikmiş gradientlerine göre parametreleri 
günceller ve optimize sürecinin daha hızlı converge olmasını sağlar.


 RMSprop: RMSprop, derin öğrenmede gradient descent optimize etme sürecinin
hızlandırılması için kullanılan bir optimize edici tekniktir. RMSprop, 
her iterasyon sırasında güncellenen gradientin ortalama karesel hata değerine dayanır
ve bu değere göre parametreleri günceller. Böylece, sert veya düzensiz gradientlerin
etkisi azaltılır ve optimize etme süreci daha hızlı converge olur.

 Gradient Descent, makine öğrenmesi ve derin öğrenmede bir modelin parametrelerini
güncellemek için kullanılan bir optimize etme algoritmasıdır. Gradient Descent'in
temel fikri, kayıp fonksiyonunu en aza indirgemek için model parametrelerinin optimal değerlerini bulmaktır.
Gradient Descent matematiksel olarak, verilen bir kayıp fonksiyonu ve model parametreleri
için bu fonksiyonun gradientini (ya da türevi) kullanarak, parametre değerlerinin optimize edilmesidir.
Gradient, modelin loss fonksiyonundaki yerel minimumların yerini gösterir ve gradientin yönü, 
loss fonksiyonunun en hızlı azalması yönünü gösterir. Bu nedenle, gradient descent algoritması,
model parametrelerini her iterasyon sırasında, gradient yönünde bir miktar adım atarak optimize
eder ve bu süreç iterasyonlar süresince devam eder, parametre değerleri loss fonksiyonunun global
minimumunu bulana kadar.

 An example of Adam optimizer is created and assigned to the "optimizer" variable. Adam optimizer
is a popular optimizer in deep learning and combines the advantages of gradient descent and
Momentum optimization techniques.

This optimizer takes three parameters:

 lr (learning rate): determines the frequency of updating the model's weights. A high
learning rate leads to fast learning but increases the risk of overshooting. A low learning
rate leads to slow learning but reduces the risk of overshooting.

 beta_1: this parameter determines the momentum value of the Momentum optimization technique.

 beta_2: this parameter is a beta value used in the RMSprop optimization technique used for optimization.

 This optimizer effectively optimizes the model's weights by combining the advantages of Momentum
and RMSprop techniques and ensures a fast and accurate convergence.

 Momentum: Momentum is an optimization technique used to accelerate the gradient descent optimization
process. The basic idea of Momentum is that if an object moves in a certain direction, even if the
gradient changes, it is inclined to move in that direction. In deep learning, the Momentum optimization
technique updates the parameters based on the accumulated gradients of past iterations and ensures that
the optimization process converges faster.

 RMSprop: RMSprop is an optimization technique used to accelerate the gradient descent optimization
process in deep learning. RMSprop is based on the average square error value of the updated gradient
in each iteration and updates the parameters accordingly. This reduces the impact of sharp or irregular 
gradients and the optimization process converges faster.

 Gradient Descent is an optimization algorithm used to update the parameters of a model in machine learning
and deep learning. The basic idea of Gradient Descent is to find the optimal values of the model parameters
to minimize the loss function. Mathematically, Gradient Descent is the optimization of the parameter values
using the gradient (or derivative) of the given loss function and model parameters. The gradient indicates
the location of the local minima in the loss function and the direction of the gradient shows the direction
of the fastest decrease of the loss function. Therefore, the gradient descent algorithm optimizes the model
parameters by taking a step in the direction of the gradient in each iteration and continues this process
until the parameter values reach the global minimum of the loss function.
"""

#%%
# Compile the model loss = "categorical_crossentropy"= birden fazla kategori binaryde 2 
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"] )

#%%
epochs = 10  # for better result increase the epochs
batch_size = 250

#%%
# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 10%
        width_shift_range=0.1,  # randomly shift images horizontally 10%
        height_shift_range=0.1,  # randomly shift images vertically 10%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)



#%%

# Fit the model
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)

#%%
# Plot the loss and accuracy curves for training and validation 
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = px.line(x=history.epoch,
              y=history.history['val_loss'],
              markers=True,
              title="Test Loss",
              color=history.history['val_loss'],
              labels={'x':'Number of Epochs', 'y':'Loss'})
fig.show()

#%%
# confusion matrix
import seaborn as sns
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
#%%
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score,confusion_matrix,mean_squared_error, r2_score,mean_absolute_error
   
print("RMS: %r " % np.sqrt(np.mean((Y_pred_classes - Y_true) ** 2)))  
print("RMSE",mean_squared_error(Y_true, Y_pred_classes,squared=False))
print("MSE",mean_squared_error(Y_true, Y_pred_classes))
#%%
scores_val = model.evaluate(X_val, Y_val, verbose=0)
scores_train = model.evaluate(X_train, Y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores_val[1]*100))
print("Accuracy: %.2f%%" % (scores_train[1]*100))
#%%

fig = px.imshow(confusion_mtx,text_auto=True, aspect="auto",
                color_continuous_scale="Viridis"
                ,labels={'x':'Predicted Label', 'y':'True Label'})
fig.show("browser")


