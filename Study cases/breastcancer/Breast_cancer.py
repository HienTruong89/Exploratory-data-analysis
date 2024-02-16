import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
print(cancer.DESCR)
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
df.describe()
df.isnull().sum()

feature=cancer['feature_names']
df_error=df[feature[10:19]]

target_names=cancer['target_names']
target=cancer['target']
df['target']=target
# Standadized data and check correlation
def Norm(X):
    X_min=X.min()
    X_max=X.max()
    Norm=(X-X_min)/(X_max - X_min)
    return Norm

X=np.array(df)
X=Norm(X)
X=pd.DataFrame(data=X,columns=df.columns)
X.head(n=5)
# Correlation - Heatmap
check=X.corr()
fig,ax=plt.subplots(figsize=(10,12))
sns.heatmap(check,vmin=-1,vmax=1)
plt.show()
# Box plot 
fig,ax=plt.subplots(figsize=(10,12))
sns.boxplot(X)
ax.tick_params(labelsize=11)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()
# Remove the extreme outliers
outliers=np.where(X['worst area']>0.9)
X_update=np.delete(X.values,outliers,axis=0)
X_update=pd.DataFrame(data=X_update,columns=X.columns)
## checking engineering data 
feature_error=['radius error','texture error','perimeter error','area error',
               'smoothness error','compactness error','concavity error',
               'concave points error','symmetry error','fractal dimension error']
data=X_update.drop(feature_error,axis=1)
data=data.drop(['target'],axis=1)
data_cor=data.corr()

fig,ax=plt.subplots(figsize=(10,12))
sns.heatmap(data_cor,vmin=-1,vmax=1)
plt.show()

fig,ax=plt.subplots(figsize=(10,12))
sns.boxplot(data)
ax.tick_params(labelsize=11)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

# checking severe outliers
data_x=data.drop(['mean area','worst area'],axis=1)
fig, ax=plt.subplots(figsize=(10,12))
sns.boxplot(data_x)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

data_x=data.drop(['mean area','worst area'],axis=1)
fig, ax=plt.subplots(figsize=(10,12))
sns.boxplot(data_x)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

data_y=data_x.drop(['mean perimeter','mean radius','mean texture','worst perimeter',
                    'worst radius','worst texture'],axis=1)
fig, ax=plt.subplots(figsize=(10,12))
sns.boxplot(data_y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()

## Descriptive statistics-univariate analysis
data_1=data

fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.boxplot(data_1,ax=ax)
ax.tick_params(labelsize=7)

# Breast cancer: worst area > 0.15
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.histplot(data_1['mean area'],bins='auto',label='Mean Area',kde=True,palette=['red'])
sns.histplot(data_1['worst area'],bins='auto',kde=True,label='Worst Area')
ax.axvline(x=data_1['mean area'].mean(),color='black',label='mean')
ax.axvline(x=data_1['worst area'].mean(),color='red')
ax.set_xlabel('mean vs worst')
ax.legend(loc='best')
plt.show()
# Breast cancer: worst perimeter > 0.02
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.histplot(data_1['mean perimeter'],bins='auto',label='Mean Perimeter',kde=True,palette=['red'])
sns.histplot(data_1['worst perimeter'],bins='auto',kde=True,label='Worst Perimeter')
ax.axvline(x=data_1['mean perimeter'].mean(),color='black',label='mean')
ax.axvline(x=data_1['worst perimeter'].mean(),color='red')
ax.set_xlabel('mean vs worst')
ax.legend(loc='best')
plt.show()
# Breast cancer: worst compactness > 2.5*10^-5
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.histplot(data_1['mean compactness'],bins='auto',label='Mean Compactness',kde=True,palette=['red'])
sns.histplot(data_1['worst compactness'],bins='auto',kde=True,label='Worst Compactness')
ax.axvline(x=data_1['mean compactness'].mean(),color='black',label='mean')
ax.axvline(x=data_1['worst compactness'].mean(),color='red')
ax.set_xlabel('mean vs worst')
ax.legend(loc='best')
plt.show()
# Breast cancer: worst concave points > 1.2
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.histplot(data_1['mean concave points'],bins='auto',label='Mean concave points',kde=True,palette=['red'])
sns.histplot(data_1['worst concave points'],bins='auto',kde=True,label='Worst concave points')
ax.axvline(x=data_1['mean concave points'].mean(),color='black',label='mean')
ax.axvline(x=data_1['worst concave points'].mean(),color='red')
ax.set_xlabel('mean vs worst')
ax.legend(loc='best')
plt.show()

## check the skewness and Kurtosis
data_1['mean area'].skew()
data_1['worst area'].kurtosis() # Kurtosis <3 --> assume normal distribution
### violin plot and swarmplot for distribution
data_1['target']=X_update['target']

fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.violinplot(data_1,x='target',y='mean area',inner=None)
sns.swarmplot(data_1,x='target',y='mean area',c='w')
ax.set_xticklabels(['Maglinant','Begnin'])
ax.set_ylim([0,0.7])
plt.show()

fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.violinplot(data_1,x='target',y='worst area',inner=None)
sns.swarmplot(data_1,x='target',y='worst area',c='w')
ax.set_xticklabels(['Maglinant','Begnin'])
ax.set_ylim([0,1])
plt.show()

fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.violinplot(data_1,x='target',y='mean fractal dimension',inner=None)
sns.swarmplot(data_1,x='target',y='mean fractal dimension',c='w')
ax.set_xticklabels(['Maglinant','Begnin'])
ax.set_ylim([1*1e-5,2.6*1e-5])
plt.show()

fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.violinplot(data_1,x='target',y='worst concave points',inner=None)
sns.swarmplot(data_1,x='target',y='worst concave points',c='w')
ax.set_xticklabels(['Maglinant','Begnin'])
ax.set_ylim([0,8*1e-5])
plt.show()
# CI the mean population for a variable: worst concavity
target_stats=data_1.groupby(['target'])['worst concavity'].agg(['mean','count','std'])
CI95_high=[]
CI95_low=[]

for i in target_stats.index:
    mean,count,std = target_stats.loc[i]
    CI95_high.append(mean + 1.96*(std/math.sqrt(count)))
    CI95_low.append(mean - 1.96*(std/math.sqrt(count)))
target_stats['CI95_high']= CI95_high
target_stats['CI95_low'] = CI95_low

# detect outliers from one variable: worst concavity
wc=data_1['worst concavity'].tail()
quantile_25=data_1['worst concavity'].quantile(0.25)
quantile_75=data_1['worst concavity'].quantile(0.75)
IQR_wc=quantile_75 - quantile_25
upper_wc=quantile_75+1.5*IQR_wc
lower_wc=quantile_25-1.5*IQR_wc
# Remove potential outliers in numpy
outliers_removed=np.where((data_1['worst concavity']> upper_wc) | (data_1['worst concavity']<lower_wc))
data_2=np.delete(data_1.values,outliers_removed,axis=0)

## Dimensional reduction with PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
sns.color_palette()

x=np.array(data)
x_pre=StandardScaler().fit_transform(x)
class_target=data_1['target']
le=LabelEncoder()
labels=le.fit_transform(class_target)

variances=[]
for i in np.arange(1,10,1):
    pca=PCA(n_components=i)
    pca.fit(x_pre)
    variances.append(np.sum(pca.explained_variance_ratio_))
plt.plot(variances)

pca=PCA(n_components=6)
scores=pca.fit_transform(x_pre)
scores_plot=pd.DataFrame(scores,columns=['PC1','PC2','PC3','PC4','PC5','PC6'])
scores_plot['target']=labels

# Set the color 
custom_pallate= {0: 'orange', 1: 'blue'}

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(18,10))
sns.scatterplot(scores_plot,x='PC1',y='PC2',hue='target',palette=custom_pallate,s=50,ax=ax1)
ax1.set_title('PC1 vs PC2',pad=20)

sns.scatterplot(scores_plot,x='PC1',y='PC3',hue='target',palette=custom_pallate,s=50,ax=ax2)
ax2.set_title('PC1 vs PC3',pad=20)

sns.scatterplot(scores_plot,x='PC1',y='PC4',hue='target',palette=custom_pallate,s=50,ax=ax3)
ax3.set_title('PC1 vs PC4',pad=20)

sns.scatterplot(scores_plot,x='PC1',y='PC6',hue='target',palette=custom_pallate,s=50,ax=ax4)
ax4.set_title('PC1 vs PC6',pad=20)

plt.tight_layout()
#plt.legend(title='Target', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

# Plot 3D PCA score
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure(figsize=(10.5,7.5))
ax=fig.add_subplot(111,projection='3d')
classes=scores_plot['target'].unique()

for c in classes:
    temp=scores_plot[scores_plot['target']==c]
    ax.scatter(temp['PC1'],temp['PC2'],temp['PC3'],label=c,s=80,color=custom_pallate[c])
ax.set_xlabel('PC1',fontsize=10)
ax.set_ylabel('PC2',fontsize=10)
ax.set_zlabel('PC3',fontsize=10)
ax.legend(title='Target')
ax.set_title('3D PCA score plot')
ax.view_init(elev=5,azim=45) ## set the view angle with elevation and azimuth
ax.zaxis.labelpad = 5
plt.show()

# Plot PCA loadings 
loadings=pca.components_.T
loadings=pd.DataFrame(loadings,columns=['PC1','PC2','PC3','PC4','PC5','PC6'])
loadings['feature']=data_1.drop('target',axis=1).columns
ax=sns.scatterplot(loadings,x='PC1',y='PC2',hue='feature',s=50)
ax.legend(loc='best', bbox_to_anchor=(1.02, 1),ncols=1,fontsize=7.5)
w=np.array(loadings['PC1'])
ax.axhline(y=0,color='black',linestyle='--',linewidth=1.0)
plt.show()

## Test if drop some minor variables would be better
data_2=data_1.drop(['mean symmetry','worst symmetry'],axis=1)
X2=np.array(data_2)
X2_pre=StandardScaler().fit_transform(X2)

variances1=[]
for i in np.arange(1,10,1):
    pca=PCA(n_components=i)
    pca.fit(X2_pre)
    variances1.append(np.sum(pca.explained_variance_ratio_))
plt.plot(variances1)

pca=PCA(n_components=5)
scores1=pca.fit_transform(X2_pre)
scores_plot1=pd.DataFrame(scores1,columns=['PC1','PC2','PC3','PC4','PC5'])
scores_plot1['target']=labels

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2,2,figsize=(18,10))
sns.scatterplot(scores_plot1,x='PC1',y='PC2',hue='target',palette=custom_pallate,s=50,ax=ax1)
ax1.set_title('PC1 vs PC2',pad=20)

sns.scatterplot(scores_plot1,x='PC1',y='PC3',hue='target',palette=custom_pallate,s=50,ax=ax2)
ax2.set_title('PC1 vs PC3',pad=20)

sns.scatterplot(scores_plot1,x='PC1',y='PC4',hue='target',palette=custom_pallate,s=50,ax=ax3)
ax3.set_title('PC1 vs PC4',pad=20)

sns.scatterplot(scores_plot1,x='PC1',y='PC5',hue='target',palette=custom_pallate,s=50,ax=ax4)
ax4.set_title('PC1 vs PC5',pad=20)

plt.tight_layout()
#plt.legend(title='Target', bbox_to_anchor=(1, 1), loc='upper left')
plt.show()

loadings1=pca.components_.T
loadings1=pd.DataFrame(loadings1,columns=['PC1','PC2','PC3','PC4','PC5'])
loadings1['feature']=data_2.columns
ax=sns.scatterplot(loadings1,x='PC1',y='PC2',hue='feature',s=50)
ax.legend(loc='best', bbox_to_anchor=(1.02, 1),ncols=1,fontsize=10)
ax.axhline(y=0,color='black',linestyle='--',linewidth=1.0)
plt.show()

# Answer --> Keep mean symmetry and worst symmetry variables


## Multivariate classification
from sklearn.model_selection import train_test_split
data_a=data_1.drop(['target'],axis=1)
X=data_a
y=labels
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

## Logistic regression classifier
from sklearn.linear_model import LogisticRegression 
log_model=LogisticRegression(solver='liblinear').fit(x_train,y_train) 
y_log=log_model.predict(x_test)

# Calculate accuracy on the test set
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
OA_log = accuracy_score(y_test, y_log)
Precision_log=precision_score(y_test, y_log)
Recall_log=recall_score(y_test, y_log)
F1_log=f1_score(y_test, y_log)
Conf_log=confusion_matrix(y_test, y_log)
print(f"Log_accuracy: {OA_log:.2f}")
print(f"Log_precision: {Precision_log:.2f}")
print(f"Log_recall: {Recall_log:.2f}")
print(f"Log_F1: {F1_log:.2f}")
print(f"Log_matrix:\n {Conf_log}")

## K-mean clustering
from sklearn.cluster import KMeans
Km_model=KMeans(n_clusters=2,max_iter=1000).fit(x_train)
centroid=Km_model.cluster_centers_
y_Km=Km_model.predict(x_test)
OA_Km=accuracy_score(y_test,y_Km)
Precision_Km=precision_score(y_test, y_Km)
Recall_Km=recall_score(y_test, y_Km)
F1_Km=f1_score(y_test, y_Km)
Conf_Km=confusion_matrix(y_test, y_Km)
print(f"Km_Accuracy: {OA_Km:.2f}")
print(f"Km_Precision:{Precision_Km:.2f}")
print(f'Km_Recall:{Recall_Km:.2f}')
print(f'Km_F1:{F1_Km:.2f}')
print(f'Km_matrix:\n{Conf_Km}')

## SVM 
# SVM linear kernel
from sklearn.svm import LinearSVC
svc_linear=LinearSVC(penalty='l1',max_iter=1000,dual=False)
l_svc_model=svc_linear.fit(x_train,y_train)
y_lsvc=l_svc_model.predict(x_test)
OA_lsvc=accuracy_score(y_test,y_lsvc)
Precision_lsvc=precision_score(y_test, y_lsvc)
Recall_lsvc=recall_score(y_test, y_lsvc)
F1_lsvc=f1_score(y_test, y_lsvc)
Conf_lsvc=confusion_matrix(y_test, y_lsvc)
print(f"l_SVC_Accuracy: {OA_lsvc:.2f}")
print(f"l_SVC_Precision:{Precision_lsvc:.2f}")
print(f'l_SVC_Recall:{Recall_lsvc:.2f}')
print(f'l_SVC_F1:{F1_lsvc:.2f}')
print(f'l_SVC_matrix:\n{Conf_lsvc}')







