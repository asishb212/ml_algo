"""
DBScan clustering
useful when there are outliers and clusters within clusters
this is not a partition algorithm
"""
import numpy as np 
import pandas as pd
import scipy as sc 
import matplotlib.pyplot as plt 
import sklearn.cluster as scl 
import sklearn.metrics as smt 
import sklearn.preprocessing as spr 
import pylab
path="/home/ashifer/code/datasets/weather-stations.csv"
df=pd.read_csv(path)
print(df.head())
df = df[pd.notnull(df["Tm"])] #dropping Nans
df = df.reset_index(drop=True)
print(df.head())
#Clustering of stations based on their location, mean, max, and min temp
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
sklearn.utils.check_random_state(1000)
Clus_dataSet = df[['Lat','Long','Tx','Tm','Tn']]
Clus_dataSet = np.nan_to_num(Clus_dataSet)
Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(Clus_dataSet)
labels = db.labels_
df["Clus_Db"]=labels
print(df[["Stn_Name","Tx","Tm","Clus_Db"]].head(5))