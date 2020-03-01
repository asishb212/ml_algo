"""
hierarchial clustering->
this is a process in which neach node in a tree is a cluster
aglomerative clustering->start from closest node and reach extreme node
desecive clustering start from farthes and reach smallest
in aglomerative clustering you find lest distances and start pairing
after each pair columns and rows are merged
this gives final dendogram

Algorithm->
1]create n clusters
2]compute distance matrix(distances between each cluster)
3]merge clusters with least distance
4]update distance matrix
5]do this till only one cluster remains

Distances between centroids->
single linkage distance->least distance between nodes
complete linkage distance->maximum distance between nodes
average linkage distance->average distance between nodes
centroid linking distance->distance between centroids
pros->
easy implementation
no need to specify number of clusters
cons->
long runtimes
can never undo previous steps
"""
import numpy as np 
import pandas as pd
import scipy as sc 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs
import sklearn.preprocessing as spr 
import pylab
from scipy.cluster.hierarchy import dendrogram
#creating dataset
path="/home/ashifer/code/datasets/cars_clus.csv"
df=pd.read_csv(path)
print(df.head())
print(df.columns)
#data cleaning
df[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = df[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.reset_index(drop=True)
features= df[['engine_s',  'horsepow', 'wheelbas', 'width', 
                    'length', 'curb_wgt', 'fuel_cap', 'mpg']]
x_data=spr.MinMaxScaler().fit_transform(features)
print(x_data[0:5])

#using scipy
indx=x_data.shape[0]
dst_mtr=np.zeros([indx,indx]) #dist matr created
for i in range(indx):
    for j in range(indx):
        dst_mtr[i,j]=sc.spatial.distance.euclidean(x_data[i],x_data[j])
#distance matrix computed
#now finding dandograms
z=sc.cluster.hierarchy.linkage(dst_mtr,'complete')
fig = pylab.figure(figsize=(18,50))
def lbls(id):
    return '[%s,%s]' % (df['manufact'][id], df['model'][id])
graph=dendrogram(z,leaf_label_func=lbls,orientation='right',leaf_rotation=0)
plt.show()