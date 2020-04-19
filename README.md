# COVID-19-Predictions-and-Financial-Crisis
This project is a work in progress but is meant to be a time series analysis and prediction of the spread of COVID-19 and the financial impact it has had. 

### Summary:
The below interactive plot shows the spread of the virus over the past few months. (last updated 3/20/2020)

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_WorldMap.PNG)

Future deaths, recoveries and confirmed cases were forecasted based on current trends and modeled. Below is a graph of the forecasted confirmed cases.

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_forecast_confirmed.png)

Stock prices have taken a dive as we see here with large tech companies but are forecasted to return to their previous positions. Some industries may never recover.

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_forecast.PNG)


### For a more indepth look at the project and findings, see below:

### Contents:
1. Data Exploration and Cleaning
2. Case Review Summary Sentiment Analysis
3. Time Series Data Exploration and Forecasting
4. Financial Data Analysis and Forecasting
5. Future Work

### Data Exploration and Cleaning:
All the data files can be found in the "Data" folder.
The initial data cleaning and exploration can be found in `Covid-19 People & Symptom Analysis Practicum.ipynb`in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.


Dataset: (https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
1. This dataset contains time series data on the number of confirmed, deaths and recovered COVID-19 cases. After some general cleaning on the "line list" data looks like this:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_data.PNG)


2. Looking for correlations in the numerical variables I ran a pair plot:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_pairwise.png)

We can see that the virus infects all age ranges but almost all deaths are older individuals:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_age_death.PNG)


3. Further cleaning of the "line list" data specifically looking at the patient summary we have:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_patient_summary.PNG)


4. Next, I cleaned the summaries by removing punctuation and digits. I left the pronouns, lemmatized the rest to remove the stop words and then joined them back together. 
A wordcloud of the results is below:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_wordcloud.png)

The wordcloud reveals that the top words are confirm, covid, patient, new, symptom, male, onset, female, wuhan, fever, pneumonia etc. We can see this further with a bar chart. 
I used matplotlib and seaborn to visualize the top occurring words post cleaning:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_topwords.png)

I visualized the top symptoms:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/line_list_topsymptoms.png)

I recreated a similar visual with interactivity in Tableau that...:
...TBD



### Case Review Summary Sentiment Analysis:
The case review summary sentiment analysis work can be found in `Covid-19 People & Symptom Analysis Practicum.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

1. In addition to previous text cleaning bi-gram and tri-gram models were made and lemmatized for the text cleaning. TextBlob was used to determine the sentiment of the summaries and the results plotted below:

...problem with final plot at the moment


### Time Series Data Exploration and Forecasting
The time series data and forecasting work can be found in `Covid-19 Time Series and Prediction Practicum.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

1. This dataset contains time series data on the number of confirmed, deaths and recovered COVID-19 cases. After some general cleaning on the "covid_19_data" file our output looks like this:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_data.PNG)

A good deal of further cleaning went into the "time_series_covid_19_confirmed", "..deaths" and "..recovered" data files as well which hold critical time series data as they track the outbreak results over-time. 


2. A plot of the reported cases by country is below:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_reported_cases.PNG)

This plot shows that China, the United States, Australia, Canada and France have the most reported cases as of the last iteration.


3. I created an interactive plotly stacked barchart that shows the total reported cases over time and the number who have recovered or died. 

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_total_stacked.png)

(insert method of linking to interactive plot outside of the jupyter notebook?)
This plot shows that the number of cases is increasing greatly and is up to over 350k but many patients recover.


4. An interactive geo scatter plot using plotly depicts the top reported cases overlaid on their countries and sized by the number of total cases:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_geoscatter.PNG)

This plot shows that China has the most reported cases at 81k as of the last iteration.

5. A similar plot in Tableau.

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/Tableau_TotalCases.PNG)

6. An interactive geo scatter plot using plotly depicts the top reported cases overlaid on their countries and sized by the number of total cases with the addition of showing the outbreak spread over time:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_WorldMap.PNG)


7. Forecasts were made using FbProphet to model the virus's upcoming outlook, the confirmed cases forecast is below:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_forecast_confirmed.png)

The deaths forecast:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_forecast_deaths.png)

The recovered cases forecast:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_forecast_recovered.png)


8. The final step was to forecast the next five days of new cases, deaths and recoveries for each country, region, state and province. Again, using Fbprophet and a couple of loops we are able to model each location and combine them back into one forecast. A portion of the results for can be seen below showing the forecast for New South Wales in Australia. 

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/time_series_forecast_results.PNG)

The Mean Absolute Error for this prediction was 70.9.


### Financial Data Analysis and Forecasting
The financial data analysis and forecasting work can be found in `Yahoo Finance API Data Practicum.ipynb` in the `Jupyter Notebooks` folder as well as the respective python files in the `Python files` folder.

1. This notebook accessed the Yahoo Finance Data API which contains time series data on company stocks. The Google data for the last five years looks like this:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_google_data.PNG)

A plot of this five year data:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_google_data.PNG)

We can see an upward trend over the past five years in Google stock and then a significant dip over the last couple months likely due to COVID-19.

Google stock five-year returns:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_google_5year_returns.PNG)

Google stock returns have spiked negatively the most in the past five years down to -10% during the COVID-19 outbreak and appears quite volatile.

2. To examine if this is a trend across large technology companies other major companies stock information is brought in from Yahoo Finance API:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_data.PNG)


A correlation plot shows similarity between the companies:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_correlation.PNG)


![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_correlation_plot.PNG)

A scatter of the five years of Google and Microsoft stocks shows slightly above average returns with more high return days than low. 

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_data.PNG)


3. The same data but only focusing on 2020. Google stocks this year:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_google_6month.PNG)

The stock has greatly declined since February. 

Google returns this year:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_google_6month_returns.PNG)


We can see that most days have a negative return since the end of February. 

Comparing Google and Microsoft only during the past 6 months we see many negative returns and some that are quite high including -15%.

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_google_vs_microsoft_6month.PNG)

Correlations between the companies are even higher now suggesting an across the board decline:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_correlation_plot_6month.PNG)

4. Forecasting major tech companies financial capability:

Past five years expected returns:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_scatter_plot.PNG)

GE and IBM have negative expected returns while the other major tech companies are positive with Microsoft having the highest expected return. (Possibly due to being awarded the JEDI contract.)

Just 2020 data expected returns:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_scatter_plot_6month.PNG)

We can see here that only Microsoft has positive expected returns and low risk while each of the other major tech companies have expected losses. 

Finally, we have the tech companies forecasted stock prices:

![alt text](https://github.com/NerdParker/COVID-19-Predictions-and-Financial-Crisis/blob/master/Images/yahoo_finance_techcomp_forecast.PNG)

The stocks are expected to recover. I did not run another forecast on just this years data as I don't believe it is enough to forecast on. I also suspect it would not be trustworthy as this would likely not suggest recovery being possible but logically if COVID-19 eventually allows for business to resume as usual the market should begin an upward trend again. 



### Future Work
Additional visualizations and dashboards. 
Financial data for other industries.
Other Forecasting methods.


