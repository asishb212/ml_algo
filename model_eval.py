import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
path="/home/ashifer/code/datasets/auto/auto_edit_one.csv"
df=pd.read_csv(path)
#regression plots
#shows scattered and regression line
sb.regplot(x="highway-mpg",y="price",data=df)
plt.show()
"""
residual plots
these plots show difference between predicted value and true value
if residual is randomly separated then model is appropriate
if it is curve then it is not a good model(variance is curvy)
if variance is increasing continuosly model is not good
"""
sb.residplot(x="highway-mpg",y="price",data=df)
plt.show()
"""
distributed plot 
these plots give predicted value vs actual value
"""
lm=LinearRegression()
"""
multiple linear regression
more than one independent variable
y=b0+b1*x1+b2*x2..........
"""
z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
lm.fit(z,df["price"])
val=lm.predict(z)
"""
distribution plots
her all nearly equally points are taken into account and pltted as histograms then plots a neat
curve from them.
"""
#plot without hist
sb.distplot(df['price'], hist=False, color="r", label="Actual Value")
sb.distplot(val, hist=False, color="b", label="Fitted Values")
plt.show()
#plot with hists
ax1 = sb.distplot(df['price'],color="r", label="Actual Value")
sb.distplot(val, color="b", label="Fitted Values" , ax=ax1)
plt.show()
#you can find r^2 or mean square error for better analysis