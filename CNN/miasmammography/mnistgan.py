# -*- coding: utf-8 -*-
"""
Created on Wed May 24 22:59:11 2023

@author: Tolga
"""

### GAN uygulama
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.datasets import mnist
from tqdm import tqdm
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam

## VERI YUKLEME YAPILIYOR
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    
    # convert shape of x_train from (60000, 28, 28) to (60000, 784) 
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return (x_train, y_train, x_test, y_test)

# (X_train, y_train,X_test, y_test)=load_data()
# print(X_train.shape)

### OPTMIZER OLARAK ADAM TANIMLANMIS-MEMORY EFFICIENCY DUSUK O SEBEPLE TERCIH ETMIS
## Adam is a combination of Adagrad and RMSprop.

def adam_optimizer():
    return Adam(lr=0.0001, beta_1=0.5)

### GENERATOR -SAHTE VERI URETECEK
def create_generator():
    generator=Sequential()
    generator.add(Dense(256,input_dim=100))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))
    
    generator.add(Dense(784, activation='tanh'))
    
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

g=create_generator() # generator modelini oluşturuyoruz
g.summary() # ekranda modeli yazdırır

## SAHTE VERIYI AYIRT EDECEK REAL/FAKE
def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(1024,input_dim=784)) # mnist görüntülerinin height x weight
    discriminator.add(LeakyReLU(0.2))    
    discriminator.add(Dropout(0.3)) # ezberlemeyi engellemek için kullanılır
    
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
       
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    
    discriminator.add(Dense(1, activation='sigmoid')) #çıktı katmanı, verinin gerçek-sahte olup olmadığına karar verir
    
    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator

d =create_discriminator()
d.summary()

## GAN NETWORK OLUSTURULUYOR
def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,)) # z: gürültü vektörü
    x = generator(gan_input) # görüntü üretimi yapılıyor
    gan_output= discriminator(x) # görüntünün sahte yada gerçek olup olmadığı sınıflandırıyor
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return gan

gan = create_gan(d,g)
gan.summary()


## OLUSTURULAN GORUNTULERI CIZIYOR
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    height,width=28,28
    generated_images = generated_images.reshape(100,height,width)
    plt.figure(figsize=figsize)
    
    # her 20 adimda uretilen goruntuleri gosteriyor
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)
    return generated_images

# GAN modelinin eğitimini gerçekleştiren kod...
def training(epochs=1, batch_size=128):    
    #Loading the data
    (X_train, y_train, X_test, y_test) = load_data()
    batch_count = X_train.shape[0] / batch_size
    
    # Creating GAN
    generator= create_generator() #generator modeli
    discriminator= create_discriminator()# discriminator modeli
    gan = create_gan(discriminator, generator) # G ve D yi birleştiren model
    
    for e in range(1,epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            #generator.trainable=False
            generated_images = generator(noise,training=False)
            
            # Get a random set of  real images
            image_batch =X_train[np.random.randint(low=0,high=X_train.shape[0],size=batch_size)]
            
            #Construct different batches of  real and fake data 
            X= np.concatenate([image_batch, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            #generator.trainable=True
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
        if e == 1 or e % 20 == 0:
           
            generated_images=plot_generated_images(e, generator)
            
training(1000,32)