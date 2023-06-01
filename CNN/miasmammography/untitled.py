# -*- coding: utf-8 -*-
"""
Created on Tue May 16 11:14:00 2023

@author: Tolga
"""

import scipy
import pandas as pd
import cv2
import matplotlib.pyplot as plt 
from skimage.color import rgb2gray
from skimage import filters
import numpy as np
import cv2
import scipy
import pandas as pd
import glob
import os
import numpy as np
from skimage.feature import greycomatrix, greycoprops
#%%
def show(img, title = 'Gray Image', rgb = False, fs = 12, dp = (10, 10)):
    plt.rcParams.update({'font.size': fs})
    plt.rcParams['figure.figsize'] = dp
    if rgb:
        plt.imshow(img[:,:,::-1])
    else:
        plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(f'{title}')
    plt.show()
def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels+1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)
        
    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask
#%%
img_mias = cv2.imread('mias/all-mias/mdb069.pgm')
img_miasp = cv2.imread('image_processed2/mdb069.pgm')
img_miasp1 = cv2.imread('image_processed/mdb069.pgm')
img_miasp1 = rgb2gray(img_miasp1)
show(img_mias)
show(img_miasp)
show(img_miasp1)

#%%
img_gray_misas = rgb2gray(img_mias)
img_gray_misas.shape
plt.imshow(img_gray_misas)


#%%

threshold = filters.threshold_sauvola(img_gray_misas)
bin_mias = (img_gray_misas > threshold)*1
kernel = np.ones((5, 5), np.uint8)
bin_mias = bin_mias.astype('uint8')
bin_mias = cv2.erode(bin_mias, kernel, iterations=-2)
show(bin_mias)

#%%
threshold = filters.threshold_isodata(img_gray_misas)
bin_mias2 = (img_gray_misas > threshold)*1
kernel = np.ones((5, 5), np.uint8)
bin_mias2 = bin_mias2.astype('uint8')
bin_mias2 = cv2.erode(bin_mias2, kernel, iterations=-2)
show(bin_mias2)
#%%
img_mask_mias = get_mask_of_largest_connected_component(bin_mias)
img_mask_mias_2 = get_mask_of_largest_connected_component(bin_mias2)
show(img_mask_mias)
show(img_mask_mias_2)
#%%
farest_pixel = np.max(list(zip(*np.where(img_mask_mias == 1))), axis=0)
nearest_pixel = np.min(list(zip(*np.where(img_mask_mias == 1))), axis=0)
if(nearest_pixel[1] == 0):
    a =  img_mias[:farest_pixel[0], :farest_pixel[1]]
else:
    a =  img_mias[nearest_pixel[0]:, nearest_pixel[1]:farest_pixel[1]]
show(a)

#%%


a=rgb2gray(a)
a.shape

show(a)


#%%




import cv2
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme uygula
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    return equalized,clahe_image

# Giriş görüntüsünü yükle
img_m = cv2.imread('mias/all-mias/mdb069.pgm')

# Histogram eşitlemeyi uygula
equalized_image,clahe_image = histogram_equalization(img_m)

# Giriş ve çıktı görüntülerini göster

show(equalized_image)
show(clahe_image)

#%%



def process_image(img):
    # Resmi yükle
    

    # Eşikleme işlemi uygula
    img_gray = rgb2gray(img)
    threshold = filters.threshold_sauvola(img_gray)
    binary = (img_gray > threshold) * 1
    kernel = np.ones((5, 5), np.uint8)
    binary = binary.astype('uint8')
    binary = cv2.erode(binary, kernel, iterations=-2)

    # En büyük bağlantılı bileşeni al
    img_mask = get_mask_of_largest_connected_component(binary)

    # Resmi kırp
    farest_pixel = np.max(list(zip(*np.where(img_mask == 1))), axis=0)
    nearest_pixel = np.min(list(zip(*np.where(img_mask == 1))), axis=0)
    if (nearest_pixel[1] == 0):
        cropped_img = img[:farest_pixel[0], :farest_pixel[1]]
    else:
        cropped_img = img[nearest_pixel[0]:, nearest_pixel[1]:farest_pixel[1]]
    
    show(cropped_img)
    equalized_image = histogram_equalization(cropped_img)
    return equalized_image

proced=process_image(img_mias)
show(proced)
#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv2.imread('mias/all-mias/mdb069.pgm')
assert img is not None, "file could not be read, check with os.path.exists()"
hist,bins = np.histogram(img.flatten(),64,[0,64])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),64,[0,64], color = 'r')
plt.xlim([0,64])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
#%%
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

#%%

hist,bins = np.histogram(img2.flatten(),64,[0,64])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img2.flatten(),64,[0,64], color = 'r')
plt.xlim([0,64])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
#%%





from skimage.color import rgb2gray
img2 = rgb2gray(img_miasp) 



from skimage.feature import greycomatrix, greycoprops
img2 = img2.astype(np.uint8)
# Verilen gri tonlamalı görüntü için GLCM hesapla
glcm = greycomatrix(img2, distances=[1], angles=[0], symmetric=True, normed=True)

# Kontrast özelliğini hesapla
contrast = greycoprops(glcm, 'contrast')

# Homojenlik özelliğini hesapla
homogeneity = greycoprops(glcm, 'homogeneity')

# Enerji özelliğini hesapla
energy = greycoprops(glcm, 'energy')


print("energy:",energy,"homogeneity:",homogeneity,"contrast:",contrast)


#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras import optimizers
from keras import losses
from sklearn import metrics

rows, cols,color = x_train[0].shape
model = Sequential()#create cnn model

model.add(Conv2D(32, (3, 3), input_shape=(rows, cols, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1),input_shape=(rows,cols,1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()
#%%


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras import optimizers
from keras import losses
from sklearn import metrics

rows, cols,color = x_train[0].shape
model = Sequential()#create cnn model

model.add(Conv2D(32, (3, 3), input_shape=(rows, cols, 1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1),input_shape=(rows,cols,1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('leaky_relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('tanh'))

model.summary()

#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten
from keras import optimizers
from keras import losses
from sklearn import metrics

rows, cols,color = x_train[0].shape
model = Sequential()#create cnn model

model.add(Conv2D(64, (3, 3), input_shape=(rows, cols, 1)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='valid', strides=(1, 1),input_shape=(rows,cols,1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))













