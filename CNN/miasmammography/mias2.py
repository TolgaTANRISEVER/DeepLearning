# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:16:50 2023

@author: Tolga

"""
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
from skimage.io import imread
import matplotlib.pyplot as plt
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
#%%

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
def histogram_equalization(image):
    # Görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram eşitleme uygula
    equalized = cv2.equalizeHist(gray)
    
    return equalized
#%%

if not os.path.exists('image_processed2'):
    os.makedirs('image_processed2')

# İşleme fonksiyonu
def process_image(img_path):
    # Resmi yükle
    img = cv2.imread(img_path)
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

   
    equalized_image = histogram_equalization(cropped_img)
    return equalized_image


# Kaydetme fonksiyonu
def save_image(img, img_path):
    # Resmi pgm olarak kaydet
    cv2.imwrite(img_path, img)


# İşlenecek dosyaların bulunduğu klasör yolu
input_folder = 'mias/all-mias/'

# Kaydedilecek dosyaların bulunduğu klasör yolu
output_folder = 'image_processed2'

# Tüm dosyaları al
files = os.listdir(input_folder)

# Her dosya için işleme uygula ve kaydet
for file in files:
    # Dosya yolu oluştur
    if file.endswith('.pgm'):
        img_path = os.path.join(input_folder, file)

        # Resmi işle
        processed_img = process_image(img_path)

        # Kaydet
        output_path = os.path.join(output_folder, file.split('.')[0] + '.pgm')
        save_image(processed_img, output_path)

        print(f"{img_path} işlendi ve {output_path} kaydedildi.")





#%%

no_angles = 360
url ='image_processed2/'
def save_dictionary(path,data):
    print('saving catalog...')
    #open('u.item', encoding="utf-8")
    import json
    with open(path,'w') as outfile:
      json.dump(str(data), fp=outfile)
      # save to file:
    print(' catalog saved')
#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from skimage.io import imread
import matplotlib.pyplot as plt
r = 'mdb015 G CIRC B 595 864 68'
image = imread("image_processed2/mdb015.pgm")
a = torch.from_numpy(image)
plt.imshow(a)
plt.scatter(516, 1000-447, s=93)
 #%%   
def read_image():
    print("Reading images")
    import cv2
    info = {}
    for i in range(322):
        if i<9:
            image_name='mdb00'+str(i+1)
        elif i<99:
            image_name='mdb0'+str(i+1)
        else:
            image_name = 'mdb' + str(i+1)
        # print(image_name)
        image_address= url+image_name+'.pgm'
        #print(image_address)
        #print(image_address)
        img = cv2.imread(image_address, 0)
        # print(i)
        img = cv2.resize(img, (64,64))   #resize image
        rows, cols = img.shape
        info[image_name]={}
        for angle in range(no_angles):
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)    #Rotate 0 degree
            img_rotated = cv2.warpAffine(img, M, (cols, rows))
            info[image_name][angle]=img_rotated
    return (info)
 #%%   
def read_lable():
    print("Reading labels")
    filename = url+'Info.txt'
    text_all = open(filename).read()
    #print(text_all)
    lines=text_all.split('\n')
    info={}
    for line in lines:
        words=line.split(' ')
        if len(words)>3:
            if (words[3] == 'B'):
                info[words[0]] = {}
                for angle in range(no_angles):
                    info[words[0]][angle] = 0
            if (words[3] == 'M'):
                info[words[0]] = {}
                for  angle in range(no_angles):
                    info[words[0]][angle] = 1
    return (info)

from sklearn.model_selection import train_test_split
import numpy as np
lable_info=read_lable()
image_info=read_image()

#%%
#print(image_info[1][0])
ids=lable_info.keys()   #ids = acceptable labeled ids
#print(type(ids))
del lable_info['Truth-Data:']
#print(lable_info)
#print(ids)
X=[]
Y=[]
for id in ids:
    for angle in range(no_angles):
        X.append(image_info[id][angle])
        Y.append(lable_info[id][angle])
X=np.array(X)
Y=np.array(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
(a,b,c)=x_train.shape  # (60000, 28, 28)
x_train = np.reshape(x_train, (a, b, c, 1))  #1 for gray scale
(a, b, c)=x_test.shape
x_test = np.reshape(x_test, (a, b, c, 1))   #1 for gray scale
# cancer_prediction_cnn(x_train, y_train, x_test, y_test)
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

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,TensorBoard
es = EarlyStopping(monitor='val_loss', mode='min', patience=5,restore_best_weights=True, verbose=1)
#%%

    

#%%


model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

file_name = 'mias_processed2'
tensorboard = TensorBoard(log_dir="logs\\{}".format(file_name))

history = model.fit(x_train, y_train,validation_split=0.2, epochs=60, batch_size=128,callbacks=[es,tensorboard])
loss_value , accuracy = model.evaluate(x_test, y_test)
#python -m tensorboard.main --logdir=logs/
print('Test_loss_value = ' +str(loss_value))
print('test_accuracy = ' + str(accuracy))

#print(model.predict(x_test))
#model.save('breast_cance_model.h5')

save_dictionary('history1.dat', history.history)
model.save_weights('proced2mias.h5')


#%% PLOTTING RESULTS (Train vs Validation)
import matplotlib.pyplot as plt
def Train_Val_Plot(acc,val_acc,loss,val_loss):
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize= (15,10))
    fig.suptitle(" MODEL'S METRICS VISUALIZATION ")

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend(['training', 'validation'])
    plt.show()
    

Train_Val_Plot(history.history['accuracy'],history.history['val_accuracy'],
               history.history['loss'],history.history['val_loss'])

#%%
y_pred=model.predict(x_test)

y_pred_prb = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
#%%
print(y_test)
print(y_pred)
y_pred=[1 if x[0]>0.7 else 0 for x in y_pred]
print(y_pred)
#%%
# def print_performance_metrics(y_test,y_pred):
#     """
#         parameters
#         ----------
#         y_test : actual label (must be in non-one hot encoded form)
#         y_pred_test : predicted labels (must be in non-one hot encoded form, common output of predict methods of classifers)

#         returns
#         -------
#         prints the accuracy, precision, recall, F1 score, ROC AUC score, Cohen Kappa Score, Matthews Corrcoef and classification report   
    
#     """
from sklearn import metrics
target=["B","M"]
print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred),4))
print('Precision:', np.round(metrics.precision_score(y_test, y_pred, average='weighted'),4))
print('Recall:', np.round(metrics.recall_score(y_test,y_pred, average='weighted'),4))
print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='weighted'),4))
print('ROC AUC Score:', np.round(metrics.roc_auc_score(y_test, y_pred_prb,multi_class='ovo', average='weighted'),4))
print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test, y_pred),4))
print('\t\tClassification Report:\n', metrics.classification_report(y_test, y_pred,target_names=target))

#%%
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt="d")

#%%

img_mias = cv2.imread('mias/all-mias/mdb069.pgm')
show(img_mias)
img_gray_misas = rgb2gray(img_mias) 
show(img_gray_misas)
print(img_gray_misas.shape)
from skimage.feature import greycomatrix, greycoprops
img_gray_misas_uint = img_gray_misas.astype(np.uint8)
# Verilen gri tonlamalı görüntü için GLCM hesapla
glcm = greycomatrix(img_gray_misas_uint, distances=[1], angles=[0], symmetric=True, normed=True)

# Kontrast özelliğini hesapla
contrast = greycoprops(glcm, 'contrast')

# Homojenlik özelliğini hesapla
homogeneity = greycoprops(glcm, 'homogeneity')

# Enerji özelliğini hesapla
energy = greycoprops(glcm, 'energy')


print("energy:",energy,"homogeneity:",homogeneity,"contrast:",contrast)
#%%
energy: [[1.]] homogeneity: [[1.]] contrast: [[0.]]
energy: [[0.72501224]] homogeneity: [[0.99589207]] contrast: [[0.00821586]]
#%%


image = imread("image_processed2/mdb069.pgm")
b = torch.from_numpy(image)
plt.imshow(b)

from skimage.feature import greycomatrix, greycoprops

# Verilen gri tonlamalı görüntü için GLCM hesapla
glcm = greycomatrix(b, distances=[1], angles=[0], symmetric=True, normed=True)

# Kontrast özelliğini hesapla
contrast = greycoprops(glcm, 'contrast')

# Homojenlik özelliğini hesapla
homogeneity = greycoprops(glcm, 'homogeneity')

# Enerji özelliğini hesapla
energy = greycoprops(glcm, 'energy')


print("energy:",energy,"homogeneity:",homogeneity,"contrast:",contrast)

