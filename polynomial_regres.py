import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
plt.show()