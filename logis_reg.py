"""
Logistic regression is a classification algorithm used to assign observations to a discrete set 
of classes. Unlike linear regression which outputs continuous number values,logistic regression 
transforms its output using the logistic sigmoid function to return a probability value which can 
then be mapped to two or more discrete classes.
note->ideally use for binary classification

linear regression technique->
all the catogaries are mapped with integers starting from 0 and features
or dependent variables are present on x axis.
so we find points on lines y=0,y=1 and so on 
now we can apply linear regression aand predict values.
the prediction where y is between the the floor values is wrong 
so we go for logistic regression

logistic regression->
here we generate graphs that look like step graphs(smooth curve that jumps from one level
to other).here we fing sigmoid function to do the job

sigmoid function(logistic function)->
sigm(A*x)=1/1+exp(A*x)
A is coefficient matrix and x is variable
in sigm function when x tends to 1 fuction becomes one as exp() becomes zero
similarly when x tends to 0 function becomes zero as exp() becomes zero
now we have a neat probability distribution
now we use conditional probability to do this->
P(Y/X1,X2) prob of y when x values are given

Algorithm->
1]initialize coeffs
2]find sigmoid function
3]compare predicted and original and record its error(cost function=sigmoid-actual_func)
but usually we consider squared/2 method so negative possibilities can be omitted
or go for mean square error
4]calculate cumulative error
5]change coeffs(gradient desent)
gradient descent->process of finding local minima
find a 3d surface for different values of coeffs aginst error
now find para having least error
this is done be moving on surface such that slope decreases using partal derivative wrt x
for every value of coeffs
6]go to step 2
7]stop when accuracy is good
aim of algorithm is simply changing cost function for better results
"""

import pylab as pl
import scipy.optimize as opt
import matplotlib.pyplot as plt
import itertools 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import sklearn.preprocessing as spr 
import sklearn.model_selection as sms
import sklearn.tree as st 
import sklearn.metrics as smt
import sklearn.linear_model as lmd 
path="/home/ashifer/code/datasets/ChurnData.csv"
df=pd.read_csv(path)
features=['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
       'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
       'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
       'lninc', 'custcat', 'churn']
x_data=np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless', 'longmon', 'tollmon', 'equipmon', 'cardmon',
       'wiremon', 'longten', 'tollten', 'cardten', 'voice', 'pager',
       'internet', 'callwait', 'confer', 'ebill', 'loglong', 'logtoll',
       'lninc', 'custcat']])
y_data=np.asarray(df['churn'])
x_data=spr.StandardScaler().fit_transform(x_data)
x_train,x_test,y_train,y_test=sms.train_test_split(x_data,y_data,test_size=0.2,random_state=4)
lr=lmd.LogisticRegression(C=0.01,solver='liblinear').fit(x_train,y_train)
#Regularization (C)is applying a penalty to increasing the magnitude of parameter 
#values in order to reduce overfitting
pred=lr.predict(x_test)#normal prediction
prob_pred=lr.predict_proba(x_test) #this gives probability 
#print(pred,prob_pred) 
#accuracy
jaccar=smt.jaccard_similarity_score(y_test,pred)
print(jaccar)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, pred, labels=[1,0]))
cnf_matrix = confusion_matrix(y_test, pred, labels=[1,0])
np.set_printoptions(precision=2)
