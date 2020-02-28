#used for predicting catogarical values
"""
classification can be done by
decision trees
naive bias
linear discriminent analysis
k-nearest neighbor
logistic regression
neural networks
support vector machines

mechanism->
by given data we can obtain a scatter plot
then we find nearest point for our values which are to predicted
but taking only one is not perfect as outliers can ruin prediction
so we consider 'k' nearest neighbours 
this KNN or k-nearest neighbour algorithm

algorithm->
1] pick a value for k
2] find distance form neighbours
    this is obtained by euclidian distance
    for example vars are x1,x2,x3 and so on
    then d=sqrt(diff.x1^2+diff.x2^2+diff.x3^2)
k is calculated by running training with differnt k and evaluating them
1]jakkerd index
jakkerd index= y ∩ ypred / y U ypred
higher j shows higher accuracy

F1 score-> obtained from confusion plots
higher j shows higher accuracy

log loss gives prob of failure
so it should be less for a good model
"""

import itertools as itl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import sklearn.preprocessing as spr 
import sklearn.model_selection as sms
import sklearn.neighbors as snb 
import sklearn.metrics as smt
path="/home/ashifer/code/datasets/teleCust1000t.csv"
df=pd.read_csv(path)
cc=df['custcat'].value_counts()
print(cc)
cols=['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
'employ', 'retire', 'gender', 'reside', 'custcat']
x_data=df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
'employ', 'retire', 'gender', 'reside']]
y_data=df["custcat"]
x_data=spr.StandardScaler().fit_transform(x_data)
k=4 #doing for 4 neighs
x_train,x_test,y_train,y_test=sms.train_test_split(x_data,y_data,test_size=0.2,random_state=1)
neigh=snb.KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
pred=neigh.predict(x_test)
acc=smt.accuracy_score(y_test,pred)#gives jaccard
acc1=smt.f1_score(y_test,pred,average='micro')
acc2=smt.f1_score(y_test,pred,average='macro')
acc3=smt.f1_score(y_test,pred,average='weighted')
#micro macro are to be used
print(acc,acc1,acc2,acc3)
#plotting for precision
mns=[]
for ks in range(1,10):
    neigh=snb.KNeighborsClassifier(n_neighbors=ks).fit(x_train,y_train)
    pred=neigh.predict(x_test)
    mns.append(smt.accuracy_score(y_test,pred))
ar=[i for i in range(1,10)]
plt.plot(ar,mns)
plt.show()
#from graph k=8 is best val 