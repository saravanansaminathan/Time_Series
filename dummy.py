import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("/Users/saravana-6868/Desktop/STUDY/kaggle/traffic/Train_SU63ISt.csv")
data.head()
df = pd.read_csv('/Users/saravana-6868/Desktop/STUDY/kaggle/traffic/Train_SU63ISt.csv', nrows = 11856)
train=df[0:10392] 
test=df[10392:]

#Aggregating the dataset at daily level
df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
df.index = df.Timestamp 
df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
train = train.resample('D').mean() 
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 
test = test.resample('D').mean()
import statsmodels.api as sm
result = sm.tsa.stattools.adfuller(train.Count)
y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)

from sklearn.metrics import mean_squared_error
rms = np.sqrt(mean_squared_error(test.Count, y_hat_avg.SARIMA))
print(rms)
a=y_hat_avg['SARIMA']
a=a.reset_index()
a=np.asarray(a).astype(str).tolist()

for i in range(len(a)):
    print(a[i])