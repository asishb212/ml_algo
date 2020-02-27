import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
path="/home/ashifer/code/datasets/auto/auto_edit_one.csv"
df=pd.read_csv(path)
x=df["highway-mpg"]
y=df["price"]
f=np.polyfit(x,y,3)  #equal to lm.fit ()
p=np.poly1d(f)
print(p)
#numpy polyfit can only be used for one variable
#so we use preprocessing library from sklearn
plr=PolynomialFeatures(degree=2)
r=plr.fit_transform([x,y])
#gives coeffs 
print(r)
#this is not perfect because of different ranges of attributes
#so we have to go for normalization
xx=df[["highway-mpg"]]
yy=df[["price"]]
scale=StandardScaler()
scale.fit(xx,yy)
df_scaled=scale.fit_transform(xx,yy)
print(df_scaled)

#------------pipelines--------------
#we use pipelines for making code simple
l=list()
l=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=6,include_bias=False)),
 ('model',LinearRegression())]
 #all processes are given in series
p=Pipeline(l)
z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
p.fit(z,yy)
pred=p.predict(z)
sb.distplot(df['price'], hist=False, color="r", label="Actual Value")
sb.distplot(pred, hist=False, color="b", label="Fitted Values")
"""
fitting of data->
underfit good overfit
underfit is using line to represent a curve
overfit is increasing index such that data is tracked too perfectly resulting inflating curves
so fing r^2 for 4-5 orders and fix one
"""

#----ridge regression----
"""
a parameter alpha is used to multiply whole eqn to reduce data tracking
"""
ri=Ridge(alpha=0.1)
ri.fit(z,yy)
rpred=ri.predict(z)
sb.distplot(rpred, hist=False, color="g", label="ridge Fitted Values")
plt.show()
#change alpha for better results

#----------gridsearch---------
"""
hyperparameters-> parameters like alpha and degree that are to be chosen
grid search automatically iterates through them generating optimal params
"""
y_data = df['price']
x_data=df.drop('price',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000],'normalize':[True,False]}]
R=Ridge()
Grid1 = GridSearchCV(R, parameters1,cv=4)
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
BestRR=Grid1.best_estimator_
b=BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)
print(b)