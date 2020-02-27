import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt 
path="/home/ashifer/code/datasets/auto/auto_edit_one.csv"
df=pd.read_csv(path)
lm=LinearRegression()
"""
multiple linear regression
more than one independent variable
y=b0+b1*x1+b2*x2..........
"""
z=df[["horsepower","curb-weight","engine-size","highway-mpg"]]
lm.fit(z,df["price"])
val=lm.predict(z)
print(val,lm.coef_,lm.intercept_)
#lm.coef_ gives all coefficients
