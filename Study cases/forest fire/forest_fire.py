import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy
from sklearn.model_selection import train_test_split

os.chdir(r'C:\Users\Hien Thi Dieu Truong\Documents\Python modelling\Git hub\Exploratory-data-analysis\Study cases\forest fire')
df=pd.read_csv('forestfires.csv')
df.head()
## understand termilogy 
# four fire weather indices:(FFMC: fine fuel moisture code; DMC: duff moisture code;DC:drough code;ISI: initial spread index) 
# check more information at the article DOI: 10.1071/WF03066
## checking the dataa
df.isnull().sum()
df.shape
## Descriptive statistical analysis
# Step 1: Observe the data by month
df=df.sort_values(by=['month'],ascending=True,inplace=False)

df['month']=pd.Categorical(df['month'],categories=["jan","feb","mar","apr","may","june","jul","aug","sep","nov","dec"],ordered=True)
data=df.sort_values(by=['month'])
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.lineplot(data=data,x='month',y='temp',color='r')
ax.set_title("Temperature across a year",fontsize=25)
ax.set_xlabel("Month",fontsize=15)
ax.set_ylabel("Temperature",fontsize=15)
ax.tick_params(labelsize=13)

# temp and wind plot across the year
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.lineplot(data=data,x='month',y='temp',color='r',label='temp')
sns.lineplot(data=data,x='month',y='wind',color='g',label='wind')
ax.set_title("Temperature, Wind across a year",fontsize=25)
ax.set_xlabel("Month",fontsize=15)
ax.set_ylabel("Temperature & Wind", fontsize=15)
ax.tick_params(labelsize=13)
plt.legend()
plt.show()
# rain plot across the year
sns.lineplot(data=data,x='month',y='rain',label='rain')
# Temp, RH, wind and rain plot across the eyar
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.lineplot(data=data,x='month',y='temp',color='r',label='temp')
sns.lineplot(data=data,x='month',y='RH',color='b',label='RH')
sns.lineplot(data=data,x='month',y='wind',color='g',label='wind')
sns.lineplot(data=data,x='month',y='rain',color='orange',label='rain')
ax.set_title("Temperature, RH, wind & rain across a year",fontsize=25)
ax.set_xlabel("Month",fontsize=15)
ax.set_ylabel("", fontsize=15) 
ax.tick_params(labelsize=13)
plt.legend()
plt.show()
#Step 2: Observe the August & July months
day=['mon','tue','wed','thu','fri','sat','sun']
df['day']=pd.Categorical(df['day'],categories=day,ordered=True)
data2=df.sort_values(by='day')
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.lineplot(data2[data2['month']=='jul'],x='day',y='temp',label='temp')
sns.lineplot(data2[data2['month']=='jul'],x='day',y='RH',label='RH')
sns.lineplot(data2[data2['month']=='jul'],x='day',y='wind',label='wind')
sns.lineplot(data2[data2['month']=='jul'],x='day',y='rain',label='rain')
ax.set_xlabel("Day",fontsize=15)
ax.set_ylabel(" ")
ax.set_title("Weather in July",fontsize=25)
plt.legend(loc='best')
# Step 3: Observe the correlation between weather variables vs fire weather indexes
data.describe()
data3=data.drop(['X','Y','month','day'],axis=1)
data3_cor=data3.corr()
data_cor=np.array(data3_cor)

ind=np.where(data_cor > 0.6)
data_cor[ind]=1
ind=np.where(data_cor <-0.4)
data_cor[ind]=-1
ind=np.where((data_cor <0.3) & (data_cor >-0.1))
data_cor[ind]=0

data_cor=pd.DataFrame(data_cor,columns=data3_cor.columns,index=data3_cor.index)
fig,ax=plt.subplots()
sns.heatmap(data_cor,cmap='coolwarm',cbar=True)
ax.set_title("Correlation map",fontsize=20)
# Step 4: checking the fire index
sns.boxplot(data=data[['FFMC','DMC','DC','ISI']])
plt.show()

data4=data[['month','area','FFMC','DMC','DC','ISI']]
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.lineplot(data=data4,x='month',y='FFMC',label='FFMC')
sns.lineplot(data=data4,x='month',y='DMC',label='DMC')
sns.lineplot(data=data4,x='month',y='DC',label='DC')
sns.lineplot(data=data4,x='month',y='ISI',label='ISI')
ax.set_xlabel("Month",fontsize=15)
ax.set_ylabel(" ")
ax.tick_params(labelsize=13)
ax.set_title('Factors indicate potential forest fire',fontsize=20)


fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.scatterplot(data4,x='area',y='FFMC',label='FFMC')
sns.scatterplot(data4,x='area',y='DMC',label='DMC')
sns.scatterplot(data4,x='area',y='DC',label='DC')
sns.scatterplot(data4,x='area',y='ISI',label='DC')

from scipy import stats
sns.scatterplot(data,x='FFMC',y='temp',label='FFMC')
sns.scatterplot(data,x='DMC',y='temp',label='DMC')
sns.scatterplot(data,x='DC',y='temp',label='DC')

sub_data=df.loc[:,['DMC','temp']]
slope,intercept,r_value,p_value, std_err=stats.linregress(x=sub_data['DMC'],y=sub_data['temp'])
print('R2:',r_value**2)
## Futher observation on area 

df.loc[df['area']>150].count()

df_1=df.drop(df.loc[df['area']>150].index)
fig,ax=plt.subplots(figsize=(10.5,7.5))
sns.scatterplot(df_1,x='area',y='FFMC',label='FFMC')
sns.scatterplot(df_1,x='area',y='DMC',label='DMC')
sns.scatterplot(df_1,x='area',y='DC',label='DC')
sns.scatterplot(df_1,x='area',y='ISI',label='ISI')



