import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
import scipy.stats as sps 
path="/home/ashifer/code/datasets/auto/auto_edit_one.csv"
df=pd.read_csv(path)
drive_wheel_count=df["drive-wheels"].value_counts()
print(drive_wheel_count)

#----------DATA VISUALIZATION---------

"""
box plots->
used to visualize distributions of data
middle line->median
box upper end->75%
box lower end->25%
line above box->upper extreme
line below box->lower extreme
dots->outliers of data
"""
sb.boxplot(x="drive-wheels",y="price",data=df)
plt.show()
"""
scatter plots->
used for visualization of continuous plots
shows relation between two variables
predictor(independent vars)->x-axis
target(value we are trying to predict)->yaxis
"""
y=df["price"]
x=df["engine-size"]
plt.scatter(x,y)
plt.title("scatter")
plt.xlabel("engine-size")
plt.ylabel("price")
#these can be used for anyplot
plt.show()
df_tst=df[['drive-wheels','body-style','price']]
df_grp=df_tst.groupby(['drive-wheels','body-style']).mean()
print(df_grp)
#we get avg cost of car by drive-wheel and bodytype
#this is hard to read so we use pivot
df_pivot=df_grp.pivot_table(index=['drive-wheels'],columns=['body-style'])
df_pivot.replace(np.nan,0,inplace=True)
print(df_pivot)
plt.pcolor(df_pivot,cmap='RdBu')
plt.colorbar()
plt.show()

#----------CORRELATION----------

#relation between variables is observed
sb.regplot(x,y,df)
plt.ylim(0,)
plt.show()
#regression plot can be seen
#ylim(bottom,top) and xlim() are used to limit lower and upper bounds
#last graph gave a positive slope showing engine and price are linearly dependent
sb.regplot(x="highway-mpg",y="price",data=df)
plt.ylim(0,)
plt.show()
#here we can see more mpg less price from graph
"""
we use strongly correlated attributes for prediction
strength is measured by different methods
PEARSON CORRELATION
we get coefficient and p-value of a data from this
coef is close to 1->large positive relation
coef is close to -1->large negative relation
coef is close to 0->no relation
pval < 0.001 strong coef certainity
pval < 0.05 moderate certainity
pval < 0.1 weak certainity
pval > 0.1 no certainity 
we can calculate using scipy stats
""" 
coef,pval=sps.pearsonr(df['horsepower'],df['price'])
print(coef,pval)
print(df.corr()) #returns coef
sb.heatmap(df.corr(), annot = True)
plt.show()

#---------ANALYSIS OF VARIABLES(ANOVA)--------------
"""
anova is used to find correlation between catogorical variables
anova returns F-test_score and p-val
f-test->variation between same group means divided by variation within same group
p-val is confidence degree
if f is small corr is less
"""
df_anova=df[["make"],["price"]]
grpd_anova=df_anova.groupby(["make"])
