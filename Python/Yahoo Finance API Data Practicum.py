#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import math
import pandas as pd
import numpy as np
import pandas_datareader.data as data
from pandas.plotting import scatter_matrix
from pandas import Series, DataFrame
import copy

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl


# In[2]:


# setting the timeframe for our data
past = datetime.datetime(2015, 1, 1)
present = datetime.datetime(2020, 4, 25)

# Yahoo API google finance data
google = data.DataReader("GOOG", 'yahoo', start=past, end=present)
google.tail()


# In[3]:


# moving average mean of closing data for past 6 months
googleClose = google['Adj Close']
avgMean = googleClose.rolling(window=180).mean()


# In[4]:


# matplotlib plot of Google data
mpl.rc('figure', figsize=(8, 7))
mpl.__version__


style.use('ggplot')
googleClose.plot(label='GOOG')
avgMean.plot(label='avgMean')
plt.legend()


# In[5]:


# returns
returns = googleClose / googleClose.shift(1) - 1
returns.plot(label='return')


# In[6]:


# Yahoo finance API data 
df = data.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=past,end=present)
df.tail()


# In[7]:


df = data.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=past,end=present)['Adj Close']
df.tail()


# In[8]:


# correlation plot
correlation = df.pct_change()
corr = correlation.corr()
corr


# In[9]:


plt.scatter(correlation.GOOG, correlation.MSFT)
plt.xlabel('Google Returns')
plt.ylabel('Microsoft Returns')


# In[10]:


plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);


# In[11]:


plt.scatter(correlation.mean(), correlation.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(correlation.columns, correlation.mean(), correlation.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


# In[12]:


google.tail()


# In[13]:


dfFeat = google.loc[:,['Adj Close','Volume']]
dfFeat['HL_PCT'] = (google['High'] - google['Low']) / google['Close'] * 100.0
dfFeat['PCT_change'] = (google['Close'] - google['Open']) / google['Open'] * 100.0


# In[14]:


dfFeat.tail()


# In[15]:


dfFeat.fillna(value=-99999, inplace=True)
# forecasting 20% of data and predicting the AdjCllose
forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.2 * len(dfFeat)))
dfFeat['label'] = dfFeat[forecast_col].shift(forecast_out)
dfFeat_noDrop = dfFeat
dfFeat.dropna(inplace=True)
dfFeat.tail()


# In[16]:


forecast_out


# In[17]:


X = np.array(dfFeat.drop(['label'],1))
y = np.array(dfFeat['label'])
X = preprocessing.scale(X)
X_late = X[-forecast_out:]
y = np.array(dfFeat['label'])


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=2)


# In[19]:


# Linear regression
linearReg = LinearRegression()
linearReg.fit(X_train, y_train)

y_predict = linearReg.predict(X_test)
accuracy = linearReg.score(X_test, y_test)

print("Prediction Accuracy: %.1f%%" % (accuracy * 100.0))


# In[20]:


# KNN Regression
knnReg = KNeighborsRegressor(n_neighbors=3)
knnReg.fit(X_train, y_train)

y_predict = knnReg.predict(X_test)
accuracy = knnReg.score(X_test, y_test)

print("Prediction Accuracy: %.1f%%" % (accuracy * 100.0))


# In[21]:


forecastStock = linearReg.predict(X_late)
dfFeat['Forecast'] = np.nan


# In[22]:


dfFeat.tail()


# In[23]:


dfFeat.iloc[-1].name


# In[24]:


dateRecent = dfFeat.iloc[-1].name
nextCalc = dateRecent + datetime.timedelta(days=1)

for i in forecastStock:
    datePred = nextCalc
    nextCalc += datetime.timedelta(days=1)
    dfFeat.loc[nextCalc] = [np.nan for _ in range(len(dfFeat.columns)-1)]+[i]
dfFeat['Adj Close'].tail(500).plot()
dfFeat['Forecast'].tail(500).plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[25]:


# setting the timeframe for our data gathering
past = datetime.datetime(2020, 1, 1)
present = datetime.datetime(2020, 4, 25)

# Yahoo API google finance data
google = data.DataReader("GOOG", 'yahoo', start=past, end=present)
google.tail()


# In[26]:


# moving average mean of closing data for past 6 months
googleClose = google['Adj Close']
avgMean = googleClose.rolling(window=180).mean()


# In[27]:


# matplotlib plot of Google data
mpl.rc('figure', figsize=(8, 7))
mpl.__version__


style.use('ggplot')
googleClose.plot(label='GOOG')
avgMean.plot(label='avgMean')
plt.legend()


# In[28]:


# returns
returns = googleClose / googleClose.shift(1) - 1
returns.plot(label='return')


# In[29]:


df = data.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=past,end=present)
df.tail()


# In[30]:


df = data.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=past,end=present)['Adj Close']
df.tail()


# In[31]:


# correlation plot
correlation = df.pct_change()
corr = correlation.corr()
corr


# In[32]:


plt.scatter(correlation.GOOG, correlation.MSFT)
plt.xlabel('Google Returns')
plt.ylabel('Microsoft Returns')


# In[33]:


plt.imshow(corr, cmap='hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);


# In[34]:


plt.scatter(correlation.mean(), correlation.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(correlation.columns, correlation.mean(), correlation.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


# In[ ]:





# In[ ]:




