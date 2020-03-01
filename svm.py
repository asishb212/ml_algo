"""
SVM is a supervised algorithm that uses a 'separator' or a vector that separates different classes
we map data to a higher dimentional space and find separating planes
separator can also be a curve(combination of vectors)
for this to work we need linearly separable data
let's say y-x is a linearly inseparable case so we go for higher dimension to make it
linearly separble. fro example we transform y-x to y-x-x^2 now we get a curve 
in which objects are linearly separable
Higher dimesioning is known as kernaling
and hd function is called kernal function
kernals available->
linear
polynomial
rbf(radial basis function)
sigmoid
2]choosing hyperplane->
lets say tow catogaries end at plane 1 and plane 2
hyperplanes is the plane which gives maximum margin
cons->
prone to overfitting
mainly used for image,sentiment,NLP 
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
import sklearn.tree as st 
import sklearn.metrics as smt
import sklearn.linear_model as lmd 
import sklearn.svm as sks
path="/home/ashifer/code/datasets/cell_samples.csv"
df=pd.read_csv(path)
graph=df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant')
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign',ax=graph)
plt.show()
df=df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc']=df['BareNuc'].astype('int')
df['Class'] = df['Class'].astype('int')
print(df.dtypes)
x_data=np.asarray(df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 
'BlandChrom', 'NormNucl', 'Mit']])
y_data=np.asarray(df['Class'])
x_train,x_test,y_train,y_test=sms.train_test_split(x_data,y_data,test_size=0.2,random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)
#choose different modes of kernal and choose one with best score
svm_mod=sks.SVC(C=1,kernel='poly')
svm_mod.fit(x_train,y_train)
pred=svm_mod.predict(x_test)
acc=smt.accuracy_score(y_test,pred)
print(acc)
kerns=['linear', 'poly', 'rbf', 'sigmoid']
l=[]
for i in kerns:
    svm_mod=sks.SVC(C=1,kernel=i)
    svm_mod.fit(x_train,y_train)
    pred=svm_mod.predict(x_test)
    l.append(smt.accuracy_score(y_test,pred))
plt.plot(kerns,l)
plt.xlabel('kernals')
plt.ylabel('accuracy')
plt.show()
#form graph you can get max
print('maximum occurs at :')
print(kerns[l.index(max(l))])