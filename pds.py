import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
#----------DATA LOADING/READING-------------------

path='/home/ashifer/code/datasets/auto/auto.csv'
df=pd.read_csv(path,header=None) #this reads csv which do not have 
#                                 headers(attributes or column names)
print(df.head(3)) #returns first n rows of df
print(df.tail(3)) # returns last n rows
attributes = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.columns=attributes #assigns attributes removing numbs in first column
path1='/home/ashifer/code/datasets/auto/atrrb.csv'
"""
df.to_csv(path1)  #you can see new csv at path
commented 13 for smooth successive runs
pandas also support .JSON .xml .sql also just replace words after "_"""

#------DATA ANALYZING---------------

"""
datatypes in pandas->
object(equal to string),int64,float64,string,datetime etc.
"""
print(df.describe(include="all")) #returns statistical values like mean,deviation etc
#add include="all" to get all data as without it non numericals are skipped
print(df.info()) #returnd dtypes non null count etc

#--------PRE PROCESSING---------

#handling missing and corrupted values
print(df["symboling"]) #used for getting column values and manipulating them
"""
when no data is found
1->delete whole entry
2->replace values say average,frequency or other functions
"""
df.dropna(subset=["price"],axis=0,inplace=True)
"""
.dropna() is used to delete whole row or column.
axis=0 for row and axis=1 for column 
all rows or columns will be deleted
inplace is made true to write .csv file permnantly
"""
print(df.head(3))
df.replace("?",np.nan, inplace = True)
print(df.head(3))
#replace() changes values in .csv file
df["normalized-losses"].replace(np.nan,140,inplace=True)

#---------DATA FORMATING-------------

df["city-mpg"]=235/df["city-mpg"]
#all math operations are possible
df.rename(columns={"city-mpg":"city-L/100"},inplace=True)
#renaming attributes
print(df.head(3),df.info())
df["price"]=df["price"].astype("float")
#astype() is used for conversion
#NOTE:np.nan's (not a number) type is FLOAT so if column has nan int is not possible
print(df["price"].mean())
#if column is of type int or float it can perform all maths omiting NaN
print(df.info())

#--------DATA NORMALIZATION---------

"""
it is process in which ranges of data is taken for comparision
ex-if data has attributes like age and income we cant just compare them.so we normalize them
in normalization we make all values from 0 to 1
method 1->scaling
x(new)=x(old)/x(max) this gives values between 0 and 1
method 2->minmax
x(new)=x(old)-x(min)/x(max)-x(min)
method 3->
x(new)=x(old)-mean/standerd_deviation
"""
#simple scaling
df["length"]=df["length"]/df["length"].max()
print(df.head(3))
#zscore
df["length"]=(df["length"]-df["length"].mean())/df["length"].std()

#data bining creating ranges so that data can be visualized in a better way like age can be binned as
#[0-5],[6-10] etc.
df["horsepower"]=df["horsepower"].astype('float')
df.replace(np.nan,df["horsepower"].mean(), inplace = True)
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
print(df[['horsepower','horsepower-binned']].head(20))
# use value_counts() to get counts
count=df["floors"].value_counts()
print(count)