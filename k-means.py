"""
k-means is a partition based clustering algorithm
it separates data into 'n' non overlaping subsets
basically it works by finding euclidian distances
it tries minimize intra distances of a cluster and maximize inter
k-means is an iterative algorithm

algorithm->
1]k-means takes k random centers for k clusters
2]now a distance_matrix is computed for each sentence
dist_matrix contains distances between point and all centers
now each point falls into a cluster with minimum distance
but this stage has a huge amount of error because of centroid positions
this error is called SSE(sum of square of errors)
3] now we move centroids to mean position of points in that cluster
4]and start again.This continues till centroid movement stops

accuracy->
1]we can compare it with a good dataset(not feasible)

2]we can measure it by finding average distance between points in a cluster
incresing k will always decrease the error
so we find elbow point in graph(point where decrease of error shifts sharply)
"""
import pylab as pl
import scipy.optimize as opt
import matplotlib.pyplot as plt
import itertools 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.preprocessing as spr 
import sklearn.model_selection as sms
import sklearn.metrics as smt
from sklearn.cluster import KMeans
import seaborn as sb 
from mpl_toolkits.mplot3d import Axes3D
path="/home/ashifer/code/datasets/Cust_Segmentation.csv"
df=pd.read_csv(path)
df =  df.drop('Address', axis=1)
features=df[['Age', 'Edu', 'Years Employed', 'Income', 'Card Debt',
       'Other Debt', 'Defaulted', 'DebtIncomeRatio']]
features = np.nan_to_num(features)
k_mean=KMeans(init = "k-means++",n_clusters=3)
k_mean.fit_transform(features)
labl=k_mean.labels_
print(labl)
df["Clus_km"] = labl
print(df.head(5))
#checking centroids
centroids=df.groupby('Clus_km').mean()
print(centroids)
area = np.pi * ( features[:, 1])**2  
plt.scatter(features[:, 0], features[:, 3], s=area, c=labl.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(features[:, 1], features[:, 0], features[:, 3], c= labl.astype(np.float))
plt.show()