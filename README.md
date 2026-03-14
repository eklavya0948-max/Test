# Analysis of Electric Load Forecasting using Linear Regression and LSTM

# Project Overview :

This project demonstrates uses of machine learning and deep learning models to predict electrical load using meteorological ( temprature and humidity ) and temporal features ( datetime , weekday , month and season ) 

# Why is electrical load forecasting important ?

In today's world, as we shift to clean resources from traditional fuel-based resources, the use of electricity is increasing in our day-to-day lives. Whether we talk about electric vehicles, data centres, induction stoves, or industrial machinery, everything needs an electrical supply. 

This transition will only succeed if we have a stable, reliable electric grid free from outages. One of the main causes of grid failure is transformer overload when demand spikes unexpectedly.

Electrical load forecasting plays a crucial role in addressing overload issues. It helps operators to predict future load and understand distribution of load throughout the grid maintaining stable electric supply and preventing major grid failures .

# Objectives :

In this project, we are focusing on the implementation and comparison of two models :

Linear Regression ( baseline machine learning model )

LSTM ( Long Short - Term Memory )

Our goal is to evaluate prediction accuracy of electric load using meteorological and temporal data.

# Dataset

The dataset used in this study is from the Ghodshila substation located in Telengana, India. The data was obtained from Mendeley Data, an open research data repository.

Source : https://data.mendeley.com/datasets/tj54nv46hj/1

Variables in the dataset

The load is calculated from:

1. Voltage (V)

2. Current (I)

3. Power factor (pf)

Additional features:

1. Power load (kW)

2. Temperature

3. Humidity

4. Weekday / Weekend

5. Season (winter, summer, rainy)

After data cleaning, the following features were used :

1. Temprature
2. Humidity
3. Hour
4. Date and Time
5. Day of week
6. Month
7. Load

