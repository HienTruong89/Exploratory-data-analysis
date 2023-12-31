import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
target=cancer['target']
df.head()
df.describe()

## checking engineering data 
df.isnull().sum()
data=df.drop(['radius error','texture error','perimeter error','smoothness error',
             'compactness error','concavity error','concave points error','symmetry error',
             'fractal dimension error','area error'],axis=1)

data1=data.drop(['mean radius','worst radius','mean compactness','worst compactness',
                 'mean concavity','worst concavity','mean concave points','worst concave points'],axis=1)
## Descriptive statistics-univariate analysis
cols1=['mean texture','worst texture','mean perimeter','worst perimeter']
cols2=['mean smoothness','worst smoothness','mean fractal dimension','worst fractal dimension']
cols3=['mean area','worst area']

data_a=data1[cols3]
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.boxplot(data_a,ax=ax)
ax.tick_params(labelsize=7)

fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.histplot(data1['mean fractal dimension'],bins='auto',label='Mean FD',kde=True,palette=['red'])
sns.histplot(data1['worst fractal dimension'],bins='auto',kde=True,label='Worst FD')
ax.axvline(x=data1['mean fractal dimension'].mean(),color='black',label='mean')
ax.axvline(x=data1['worst fractal dimension'].mean(),color='black')
ax.set_xlabel('mean fractal dimension vs worst fractal dimension')
ax.legend(loc='best')
plt.show()
## check the skewness and Kurtosis
data1['worst fractal dimension'].skew()
data1['worst fractal dimension'].kurtosis() ## extreme likehood compared to normal distribution (k=3)
### violin plot and swarmplot for distribution 
da=data1[['mean texture','worst texture','mean perimeter','worst perimeter','mean fractal dimension',
          'worst fractal dimension']]
da['target']=pd.DataFrame(cancer['target'])
fig=plt.figure(figsize=(10.5,7.5))
sns.violinplot(data=da,x='target',y='mean fractal dimension',inner=None)
sns.swarmplot(data=da,x='target',y='mean fractal dimension',c='w')
plt.show()
### Confidence interval of the mean population from the variable: mean fractal dimension
target_stats=da.groupby(['target'])['mean fractal dimension'].agg(['mean','count','std'])
CI95_high=[]
CI95_low=[]

for i in target_stats.index:
    mean,count,std = target_stats.loc[i]
    CI95_high.append(mean + 1.96*(std/math.sqrt(count)))
    CI95_low.append(mean - 1.96*(std/math.sqrt(count)))
target_stats['CI95_high']= CI95_high
target_stats['CI95_low'] = CI95_low

cof_int_low,cof_int_high=scipy.stats.norm.interval(0.95,loc=mean,scale=std)
cof_int_low,cof_int_high
## detect outliers 
wfd=data1['worst fractal dimension'].tail()
quantile_25=data1['worst fractal dimension'].quantile(0.25)
quantile_75=data1['worst fractal dimension'].quantile(0.75)
IQR_wfd=quantile_75 - quantile_25
upper_wrd=quantile_75+1.5*IQR_wfd
lower_wrd=quantile_25-1.5*IQR_wfd
## Define outliers in a specific column using pandas
outliers=[]
for index, row in data1.iterrows():
    if (row['worst fractal dimension']> upper_wrd) or (row['worst fractal dimension']<lower_wrd):
        outlier=row['worst fractal dimension']
        outliers.append(outlier)

## Remove outliers in dataframe
outliers=[]
for index, row in data1.iterrows():
    if (row['worst fractal dimension']> upper_wrd) or (row['worst fractal dimension']<lower_wrd):
        outliers.append(index)
data1_update=data1.drop(outliers, axis=0)

# Remove outliers in numpy
data1x=np.array(data1)
outliers_removed=np.where((data1['worst fractal dimension']> upper_wrd) | (data1['worst fractal dimension']<lower_wrd))
data1_update1=np.delete(data1x,outliers_removed,axis=0)

## dimensional reduction with PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
X=np.array(data_update[:,0:12])
X_pre=StandardScaler().fit_transform(X)
class_target=data_update[:,-1]

variances=[]
for i in np.arange(1,10,1):
    pca=PCA(n_components=i)
    pca.fit(X_pre)
    variances.append(np.sum(pca.explained_variance_ratio_))
plt.plot(variances)

pca=PCA(n_components=5)
scores=pca.fit_transform(X_pre)
scores_plot=pd.DataFrame(scores,columns=['PC1','PC2','PC3','PC4','PC5'])
scores_plot['target']=class_target
sns.scatterplot(scores_plot,x='PC1',y='PC2',hue='target')
# Plot 3D PCA score
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10.5,7.5))
ax=fig.add_subplot(111,projection='3d')
classes=scores_plot['target'].unique()

for c in classes:
    temp=scores_plot[scores_plot['target']==c]
    ax.scatter(temp['PC1'],temp['PC2'],temp['PC3'],label=c,s=80)
ax.set_xlabel('PC1',fontsize=10)
ax.set_ylabel('PC2',fontsize=10)
ax.set_zlabel('PC3',fontsize=10)
ax.legend()
ax.view_init(elev=5,azim=45) ## set the view angle with elevation and azimuth
ax.zaxis.labelpad = 5
plt.show()

# plot PCA loadings 
loadings=pca.components_.T
loadings=pd.DataFrame(loadings,columns=['PC1','PC2','PC3','PC4','PC5'])
loadings['feature']=data1.drop('target',axis=1).columns
ax=sns.scatterplot(loadings,x='PC1',y='PC2',hue='feature')
ax.legend(loc='best', bbox_to_anchor=(1.02, 1),ncols=1,fontsize=10)
w=np.array(loadings['PC1'])
ax.axhline(y=0,color='black',linestyle='--',linewidth=1.0)
plt.show()

## Drop minor variables 
data1_up=data1_update.drop(['mean texture','worst texture'],axis=1)
X1=np.array(data1_up)
X_pre1=StandardScaler().fit_transform(X1)

variances1=[]
for i in np.arange(1,10,1):
    pca=PCA(n_components=i)
    pca.fit(X_pre1)
    variances1.append(np.sum(pca.explained_variance_ratio_))
plt.plot(variances1)

pca=PCA(n_components=4)
scores=pca.fit_transform(X_pre1)
scores_plot=pd.DataFrame(scores,columns=['PC1','PC2','PC3','PC4'])
scores_plot['target']=class_target
sns.scatterplot(scores_plot,x='PC1',y='PC2',hue='target')

loadings1=pca.components_.T
loadings1=pd.DataFrame(loadings1,columns=['PC1','PC2','PC3','PC4'])
loadings1['feature']=data1_up.columns
ax=sns.scatterplot(loadings1,x='PC1',y='PC2',hue='feature')
ax.legend(loc='best', bbox_to_anchor=(1.02, 1),ncols=1,fontsize=10)
ax.axhline(y=0,color='black',linestyle='--',linewidth=1.0)
plt.show()


### Inferential statictis 
## Hypothesis with ANOVA & linear model fitting
# Test one-way anova
group1=np.array(data1['mean texture'])
group2=np.array(data1['worst texture'])
f_oneway(group1,group2)


anov=[]
groups=[]
for i in data1.columns.unique():
    group=data1[i]
    groups.append(group)

for i in range(len(groups)-1):
    anova=f_oneway(groups[i].values,groups[i+1].values)
    anov.append(anova)
# pearsonr
r2=[]
for j in range(len(groups)-1):
    pr2=pearsonr(groups[j].values.reshape(-1),groups[j+1].values.reshape(-1))
    r2.append(pr2)
    
# Mutual information compared to class_target
from sklearn.feature_selection import mutual_info_classif
Sel=pd.DataFrame(mutual_info_classif(data1_up, class_target))
Sel['feature']=list(data1_up.columns)
print('Mutual information:',Sel)
### Chi-square test to compare observed results and with expected results
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

df_new=SelectKBest(chi2,k=4).fit_transform(data1_up, class_target)

from scipy import stats  
from statsmodels.stats.weightstats import DescrStatsW ## operate stats on the object

Test=DescrStatsW(data1['mean fractal dimension'])
Test.mean
q3=Test.quantile(0.75) ## pandas object
q1=Test.quantile(0.25)
IQR=q3.loc[0.75]-q1.loc[0.25]

# or 
stats.iqr(data1['mean fractal dimension']) ## interpolation if data have lower,higher point
stats.zscore(data1['mean fractal dimension']) ## calculate the propablity of each point around the mean =0, std=1
## or numpy
q1=np.percentile(data1['mean fractal dimension'], 25)
q3=np.percentile(data1['mean fractal dimension'], 75)
I=q3-q1

## Fit linear regression plot with seaborn
sns.lmplot(da,x='worst fractal dimension',y='mean fractal dimension')
#call linear regression using scipy
slope, intercept,r_value, _,_,=stats.linregress(da['worst fractal dimension'],da['mean fractal dimension'])
print('R2', r_value**2)


## Mutivariate linear regression for with  OLS 
import statsmodels.api as sm
X=da.drop(['target'],axis=1)
y=da['target']

reg_model=sm.OLS(y,X).fit()
reg_model.params
reg_model.summary()


## Multivariate classification
from sklearn.model_selection import train_test_split
X=data1_up
y=class_target
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

## Logistic regression classifier (for classification not regression)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(solver='liblinear').fit(x_train,y_train) 

y_pred=log.predict(x_test)
# Calculate accuracy on the test set
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {overall_accuracy:.2f}")

class_accuracy=[]
F1=[]
Recall=[]
Precision=[]
for i in np.unique(y_test):
    label=np.where(y_test == i)
    class_acc=accuracy_score(y_test[label],y_pred[label])
    class_accuracy.append(class_acc)
    fscore=f1_score(y_test[label],y_pred[label],average='micro')
    F1.append(fscore)
    re=recall_score(y_test[label],y_pred[label],average='micro')
    Recall.append(re)
    pre=precision_score(y_test[label],y_pred[label],average='micro')
    Precision.append(pre)


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

labels=['Magilant','Benign']
Results=pd.DataFrame({'Class accuracy':[class_accuracy],'F1':[F1],'Precision':[Precision],'Recall':[Recall]}).transpose()
Results['labels']=labels
print(Results)





