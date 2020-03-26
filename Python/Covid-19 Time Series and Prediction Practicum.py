#!/usr/bin/env python
# coding: utf-8

# In[1]:


# packages to store and manipulate data
import numpy as np
import pandas as pd
from pprint import pprint

# spacy for lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")

# packages for visualizations
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as graphO
from fbprophet import Prophet
import pycountry

# conda install -c conda-forge fbprophet


# In[2]:


# Import Time-series Dataset, Last Updated on 3/20/2020
# Kaggle COVID Datasource: https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset

# this file is the same data as in df1 but with case numbers and no lat/long
df = pd.read_csv('C:/Users/607791/Desktop/DS/Practicum II COVID-19/covid_19_data.csv',parse_dates=['Last Update'])
df.rename(columns={'ObservationDate':'Date'}, inplace=True)

# these three files hold critical time series data as they track the outbreak over-time
df_confirmed = pd.read_csv("C:/Users/607791/Desktop/DS/Practicum II COVID-19/time_series_covid_19_confirmed.csv")
df_recovered = pd.read_csv("C:/Users/607791/Desktop/DS/Practicum II COVID-19/time_series_covid_19_recovered.csv")
df_deaths = pd.read_csv("C:/Users/607791/Desktop/DS/Practicum II COVID-19/time_series_covid_19_deaths.csv")


# In[3]:


# Rename countries so they show up in pycountry
df["Country/Region"].replace({'US': 'United States'}, inplace=True)
df_confirmed["Country/Region"].replace({'US': 'United States'}, inplace=True)
df_recovered["Country/Region"].replace({'US': 'United States'}, inplace=True)
df_deaths["Country/Region"].replace({'US': 'United States'}, inplace=True)

df["Country/Region"].replace({'UK': 'United Kingdom'}, inplace=True)
df_confirmed["Country/Region"].replace({'UK': 'United Kingdom'}, inplace=True)
df_recovered["Country/Region"].replace({'UK': 'United Kingdom'}, inplace=True)
df_deaths["Country/Region"].replace({'UK': 'United Kingdom'}, inplace=True)

df["Country/Region"].replace({'Mainland China': 'China'}, inplace=True)
df_confirmed["Country/Region"].replace({'Mainland China': 'China'}, inplace=True)
df_recovered["Country/Region"].replace({'Mainland China': 'China'}, inplace=True)
df_deaths["Country/Region"].replace({'Mainland China': 'China'}, inplace=True)


# In[4]:


df_confirmed.head()


# In[5]:


df_date = df.groupby('Date').sum()


# In[6]:


plt.figure(2, figsize=(14, 14/1.5))
plt.title("Top Reported Covid-19 Countries")
df['Country/Region'].value_counts()[:25].plot('bar')


# In[7]:


confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()


# In[8]:


totalCases = graphO.Figure()
totalCases.add_trace(graphO.Bar(x = confirmed['Date'],
                y = confirmed['Confirmed'],
                name = 'Confirmed',
                marker_color = 'cyan'
                ))

totalCases.add_trace(graphO.Bar(x = recovered['Date'],
                y = recovered['Recovered'],
                name = 'Recovered',
                marker_color = 'blue'
                ))

totalCases.add_trace(graphO.Bar(x=deaths['Date'],
                y = deaths['Deaths'],
                name = 'Deaths',
                marker_color = 'magenta'
                ))

totalCases.update_traces(
    textposition = 'outside'
)

totalCases.update_layout(
    title = 'Total Reported Covid-19 Cases',
    xaxis_tickfont_size = 12,
    yaxis = dict(
        title = 'Reported Cases',
        titlefont_size = 18,
        tickfont_size = 12,
    ),
    legend = dict(
        x = 1,
        y = 1,
    ),
    barmode = 'stack',
    bargroupgap = 0.1
)
totalCases.show()


# In[9]:


df_map = df.groupby(["Date", "Country/Region", "Province/State"])[['SNo', 'Date', 'Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()


# In[10]:


confirmed = df_map.groupby(['Date', 'Country/Region']).sum()[['Confirmed']].reset_index()


# In[11]:


confirmed.head()


# In[12]:


latest_date = confirmed['Date'].max()


# In[13]:


map_confirm = confirmed[(confirmed['Date'] == latest_date)][['Country/Region', 'Confirmed']]
countryUnique = map_confirm['Country/Region'].unique()


# In[14]:


# Getting countries iso for map hover
countries = {}

for country in pycountry.countries:
    countries[country.name] = country.alpha_3
    
map_confirm["iso_alpha"] = map_confirm["Country/Region"].map(countries.get)


# In[15]:


# Plotly scatter geo guide: https://plot.ly/python/scatter-plots-on-maps/
plot_map = map_confirm[["iso_alpha","Confirmed", "Country/Region"]]
fig = px.scatter_geo(plot_map, 
                     locations = "iso_alpha", 
                     color = "Country/Region",
                     hover_name = "iso_alpha", 
                     size = "Confirmed",
                     projection = "natural earth", title = 'Covid-19 Cases')
fig.show()


# In[16]:


df_confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
df_recovered = df.groupby('Date').sum()['Recovered'].reset_index()
df_deaths = df.groupby('Date').sum()['Deaths'].reset_index()
df_confirmed.head()


# In[17]:


# fix the date
df_confirmed['Date'] = pd.to_datetime(df_confirmed['Date'])
df_confirmed.head()


# In[18]:


# columns must have these names to use Prophet to forecast
df_confirmed.columns = ['ds','y']


# In[19]:


# forecasting model at 98% prediction intervals
model = Prophet(interval_width=0.98)
model.fit(df_confirmed)

modPred = model.make_future_dataframe(periods=8)
forecastPred = model.predict(modPred)
df_confirmed_forecast = model.plot(forecastPred)


# In[20]:


# fix the date
df_deaths['Date'] = pd.to_datetime(df_deaths['Date'])
# columns must have these names to use Prophet to forecast
df_deaths.columns = ['ds','y']

# forecasting model at 98% prediction intervals
model = Prophet(interval_width=0.98)
model.fit(df_deaths)

modPred = model.make_future_dataframe(periods=8)
forecastPred = model.predict(modPred)
df_deaths_forecast = model.plot(forecastPred)


# In[21]:


# fix the date
df_recovered['Date'] = pd.to_datetime(df_recovered['Date'])
# columns must have these names to use Prophet to forecast
df_recovered.columns = ['ds','y']

# forecasting model at 98% prediction intervals
model = Prophet(interval_width=0.98)
model.fit(df_recovered)

modPred = model.make_future_dataframe(periods=8)
forecastPred = model.predict(modPred)
df_deaths_forecast = model.plot(forecastPred)


# In[22]:


df.head()


# In[23]:


forecast5days = 5
sortDate = sorted(list(set(df['Date'].values)))[-forecast5days]


# In[24]:


# using the split dataset from earlier df_map
df_confirmed = df_map[['SNo', 'Date','Province/State', 'Country/Region', 'Confirmed']]
df_deaths = df_map[['SNo', 'Date','Province/State', 'Country/Region', 'Deaths']]
df_recovered = df_map[['SNo', 'Date','Province/State', 'Country/Region', 'Recovered']]


# In[25]:


df_confirmed.head()


# In[26]:


# my own take but major credit and inspiration from khonongweihoo on Kaggle 
# as I have no prior experience with forecasting models
# https://www.kaggle.com/khoongweihao/covid-19-novel-coronavirus-eda-forecasting-cases/output

# each country with reported cases
df_forecasts = []
error = []
counter = 0
# forecast for each country with reported cases
for country in countryUnique:
    try:
        assert(country in df_confirmed['Country/Region'].values)
        df_confirmedCountryRegion = df_confirmed[(df_confirmed['Country/Region'] == country)]
        df_deathsCountryRegion = df_deaths[(df_deaths['Country/Region'] == country)]
        df_recoveredCountryRegion = df_recovered[(df_recovered['Country/Region'] == country)]
        df_countryRegion = [('Confirmed', df_confirmedCountryRegion), 
                       ('Deaths', df_deathsCountryRegion), 
                       ('Recovered', df_recoveredCountryRegion)]
        
        # each province/state 
        provinceState = df_confirmedCountryRegion['Province/State'].unique()
        # loop through each province
        for province in provinceState:
            try:
                df_provinceState = []
                
                assert(province in df_confirmedCountryRegion['Province/State'].values)
                
                # forecast for Confirmed, Deaths and Recovered
                for country_tuple in df_countryRegion:
                    CaseResult = country_tuple[0]
                    df_country = country_tuple[1]
                    df_province = df_country[(df_country['Province/State'] == province)]

                    # data preparation for forecast with Prophet at province level
                    df_province = df_province[['Date', CaseResult]]
                    df_province.columns = ['ds','y']
                    df_province['ds'] = pd.to_datetime(df_province['ds'])
                    
                    df_provinceValidation = df_province[(df_province['ds'] >= pd.to_datetime(sortDate))] # validation set
                    df_province = df_province[(df_province['ds'] < pd.to_datetime(sortDate))] # train set
                    
                    # model using Prophet just like before
                    model = Prophet()
                    model.fit(df_province)
                    modPred = model.make_future_dataframe(periods=forecast5days)
                    forecastPred = model.predict(modPred)
                    
                    # saves the mean absolute error to the error variable after evaluating the validation data
                    df_forecast = forecastPred[['ds', 'yhat']]
                    df_final = df_forecast[(df_forecast['ds'] >= pd.to_datetime(sortDate))]
                    df_finalValidation = df_final.merge(df_provinceValidation, on=['ds'])
                    df_finalValidation['abs_diff'] = (df_finalValidation['y'] - df_finalValidation['yhat']).abs()
                    error += list(df_finalValidation['abs_diff'].values)
                    
                    # output results into a data frame
                    df_forecast['Province/State'] = province
                    df_forecast['Country/Region'] = country
                    df_forecast.rename(columns={'yhat':CaseResult}, inplace=True)
                    df_provinceState += [df_forecast.tail(forecast5days)]
                
                df_merge = df_provinceState[0].merge(df_provinceState[1],on=['ds', 'Province/State', 'Country/Region']).merge(df_provinceState[2],on=['ds', 'Province/State', 'Country/Region'])
                df_forecasts += [df_merge]
            except:
                continue
    except:
        continue


# In[34]:


# concatenate the dataframes into one
df_forecastResult = pd.concat(df_forecasts, axis=0)
#sort the results
df_forecastResult.sort_values(by='ds')
# rename columns
df_forecastResult = df_forecastResult[['ds', 'Province/State', 'Country/Region', 'Confirmed', 'Deaths', 'Recovered']]
df_forecastResult.rename(columns={'ds':'Date'}, inplace=True)
#round to integers/whole numbers and make the any negative forecasts zero
for CaseResult in ['Confirmed', 'Deaths', 'Recovered']:
    df_forecastResult[CaseResult] = df_forecastResult[CaseResult].round()
    df_forecastResult[df_forecastResult[CaseResult] < 0] = 0

# five day forecast error evaluation    
errorLength = len(error)
mae = sum(error)/errorLength
print('Five day forecast mean absolute error: ' + str(round(mae, 1)))

df_forecastResult.head()


# In[ ]:




