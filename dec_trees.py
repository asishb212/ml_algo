"""
this is also a classification method
Decision tree is a method in which attributes are given hierarchies for predicting
each level has some set of attributes which are branched continously for getting final output
Choosing attribute order(the order or height at which we should place them)->
we do this by computing a factor called entropy
entropy is amount of disorder in a set
so we choose attributes that have less entropy
we find entropy after splitting
now we get 2 values for each branch obtained say e1 and e2
now we find information gain to finalize attribute
info_gain= entropy before split-weighted entropy after split
let's say splited one has 2 types of catogarical variables
weighted_entropy= (number of type1/total)*e1+(number of type2/total)*e2
now choose one with high info gain
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
import sklearn.tree as st 
import sklearn.metrics as smt
path="/home/ashifer/code/datasets/drug200.csv"
df=pd.read_csv(path)
print(df.columns)
print(df.head())
le_sex=spr.LabelEncoder()
le_BP=spr.LabelEncoder()
le_Chol=spr.LabelEncoder()
le_sex.fit(["F","M"])
df["Sex"]=le_sex.transform(df["Sex"])
le_BP.fit(["LOW","NORMAL","HIGH"])
df["BP"]=le_BP.transform(df["BP"])
le_Chol.fit(["NORMAL","HIGH"])
df["Cholesterol"]=le_Chol.transform(df["Cholesterol"])
print(df.head())
#label encoder is used to change values of catogarical attributes
x_data=df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y_data=df['Drug']
x_train,x_test,y_train,y_test=sms.train_test_split(x_data,y_data)
drugtree=st.DecisionTreeClassifier(criterion="entropy",max_depth=4)
drugtree.fit(x_train,y_train)
pred=drugtree.predict(x_test)
print(pred[0:5])
acc=smt.accuracy_score(y_test,pred)
print(acc)
#visualization is not required for trees
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=st.export_graphviz(drugtree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')