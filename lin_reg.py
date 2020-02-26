#use auto_edit_one.csv
"""
regression->simple or linear reg
          ->multi reg
simple regression-->
getting model in form of y=b0+b1*x (linear dependency)
to find this line we mark data points and use thiem to FIT our model
process=>> 1]get data 2]get coef by fiting 3]training 4]predict val
"""
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression as lsr 
import matplotlib.pyplot as plt 
lm=lsr()
path="/home/ashifer/code/datasets/auto/auto_edit_one.csv"
df=pd.read_csv(path)
x=df[["highway-mpg"]]
y=df[["price"]]
lm.fit(x,y)
pred=lm.predict(x)
#get eqns
b0=lm.intercept_
b1=lm.coef_
print(b0,b1)
#never call numpy.ndarray with parenthesis unless you want to arguments or parameters
plt.plot(x,pred)
plt.scatter(x,y)
plt.show()