# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:42:30 2023

@author: Tolga
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:33:18 2022

@author: Tolga
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#%% create dataset 
mu ,sigma=25,5
x1 = np.random.normal(mu,sigma,1000)
y1 = np.random.normal(mu,sigma,1000)
x2 = np.random.normal(55,sigma,1000)
y2 = np.random.normal(60,sigma,1000)
x3 = np.random.normal(55,sigma,1000)
y3 = np.random.normal(15,sigma,1000)
#%%
for i in x1:
    print(i)

count, bins, ignored = plt.hist(x1, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()

x=np.concatenate((x1,x2,x3),axis=0)
y=np.concatenate((y1,y2,y3),axis=0)


dic={"x":x,"y":y}
data = pd.DataFrame(dic)

plt.scatter(x1,y1,color="b")
plt.scatter(x2,y2,color="b")
plt.scatter(x3,y3,color="b")
plt.show()

#%%k-means
from sklearn.cluster import KMeans
wcss=[]

for k in range(1,15):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,15),wcss)
plt.xticks(range(1,15))
plt.xlabel("number og k (cluster) value")
plt.ylabel("wcss")
plt.show()
#%% k = 3 for model  

kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data)

data["label"]=clusters

plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="blue")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="green")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

#%%Prediction
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    data.T, 3, 2, error=0.005, maxiter=3000)
# Predict new cluster membership with `cmeans_predict` as well as
# `cntr` from the 3-cluster model
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    data.T,cntr , 3 ,error=0.005, maxiter=3000)

# Plot the classified uniform data. Note for visualization the maximum
# membership value has been taken at each point (i.e. these are hardened,
# not fuzzy results visualized) but the full fuzzy result is the output
# from cmeans_predict.
cluster_membership = np.argmax(u, axis=0)  # Hardening for visualization
data["labelFCM"]=cluster_membership


#%%Evaluating the ANN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

# Küme merkezlerini ve tahminleri al
centroids = kmeans.cluster_centers_

#Yapay Sinir Ağı (ANN) modeli
ann_model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam')
ann_model.fit(data, data.label)

# Tahminler yap
predictions = ann_model.predict(data)

# Sonuçları görselleştir
plt.scatter(data['x'], data['y'], c=predictions, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, color='red')
plt.show()
data["labelANN"]=predictions
#%%


#%%

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from scipy import optimize

# Veri setini oluşturma
mu, sigma = 25, 5
x1 = np.random.normal(mu, sigma, 1000)
y1 = np.random.normal(mu, sigma, 1000)
x2 = np.random.normal(55, sigma, 1000)
y2 = np.random.normal(60, sigma, 1000)
x3 = np.random.normal(55, sigma, 1000)
y3 = np.random.normal(15, sigma, 1000)

x = np.concatenate((x1, x2, x3), axis=0)
y = np.concatenate((y1, y2, y3), axis=0)

data1 = np.vstack((x, y))
data=data1.T

# ANFIS modelini oluşturma
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
    data, 3, 2, error=0.005, maxiter=3000)

def anfis_predict(x, y):
    def membership_functions(data1, centers):
        membership = []
        for i in range(len(centers)):
            membership.append(fuzz.membership.gaussmf(data1, centers[i], sigma))
        return membership

    def anfis_rules(membership):
        rules = []
        for i in range(len(membership[0])):
            rule = []
            for j in range(len(membership)):
                rule.append(membership[j][i])
            rules.append(rule)
        return np.array(rules)

    def premise_values(rules, x, y):
        values = []
        for rule in rules:
            value = []
            for i in range(len(rule)):
                value.append(rule[i] * x[i] * y[i])
            values.append(value)
        return np.array(values)

    def normalize_weights(weights):
        sum_weights = np.sum(weights)
        return weights / sum_weights

    def consequent_values(values):
        result = np.sum(values, axis=1)
        return normalize_weights(result)

    def predict(x, y, cntr):
        membership_x = membership_functions(x, cntr[0])
        membership_y = membership_functions(y, cntr[1])
        rules = anfis_rules([membership_x, membership_y])
        premise = premise_values(rules, x, y)
        consequent = consequent_values(premise)
        return consequent

    return predict(x, y, cntr)

# Verileri tahmin etme
predictions1 = anfis_predict(data1[0], data1[1])

# Sonuçları görselleştirme
cluster_membership = np.argmax(predictions1, axis=0)

plt.scatter(data1[0, :], data1[1, :], c=cluster_membership)
plt.show()

#%%
data["labelANFIS"]=cluster_membership
plt.scatter(data.x[data.label==0],data.y[data.label==0],color="red")
plt.scatter(data.x[data.label==1],data.y[data.label==1],color="blue")
plt.scatter(data.x[data.label==2],data.y[data.label==2],color="green")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
plt.title("data.label")
plt.show()
plt.scatter(data.x[data.labelANFIS==0],data.y[data.labelANFIS==0],color="black")
plt.scatter(data.x[data.labelANFIS==1],data.y[data.labelANFIS==1],color="brown")
plt.scatter(data.x[data.labelANFIS==2],data.y[data.labelANFIS==2],color="pink")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
plt.title("data.labelANFIS")
plt.show()
plt.scatter(data.x[data.labelANN==0],data.y[data.labelANN==0],color="orange")
plt.scatter(data.x[data.labelANN==1],data.y[data.labelANN==1],color="cyan")
plt.scatter(data.x[data.labelANN==2],data.y[data.labelANN==2],color="purple")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
plt.title("data.labelANN")
plt.show()
plt.scatter(data.x[data.labelFCM==0],data.y[data.labelFCM==0],color="gray")
plt.scatter(data.x[data.labelFCM==1],data.y[data.labelFCM==1],color="#942c35")
plt.scatter(data.x[data.labelFCM==2],data.y[data.labelFCM==2],color="#cfe643")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")
plt.title("data.labelFCM")
plt.show()

#%%
