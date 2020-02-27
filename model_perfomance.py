#here we will obtain how well our model is trained
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
path="/home/ashifer/code/datasets/auto/auto_edit_two.csv"
df=pd.read_csv(path)
y_data = df['price']
x_data=df.drop('price',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])
"""
trainings ran by different data will be different
so we chose cross validation for better result
CROSS VALIDATION->
we divide data into equal sets and use them for training ang testing
we do this till all sets acted as both test and train data
atlast we take average values 
"""
lr=LinearRegression()
scores=cross_val_score(lr,x_data[["horsepower","highway-mpg"]],y_data,cv=3)
fin_val=np.mean(scores)
print(fin_val)
#this just takes good values
pred=cross_val_predict(lr,x_data[["horsepower","highway-mpg"]],y_data,cv=3)
print(pred)
#this gives pridicted value too
lr2=LinearRegression()
lr2.fit(df[["horsepower","highway-mpg"]],df[["price"]])
pred2=lr2.predict(df[["horsepower","highway-mpg"]])
sb.distplot(df['price'], hist=False, color="r", label="Actual Value")
sb.distplot(pred, hist=False, color="b", label="Fitted Values using cross_val")
sb.distplot(pred2, hist=False, color="g", label="Fitted Values using multi_reg")
plt.show()