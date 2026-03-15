# -*- coding: utf-8 -*-

# %% [markdown] Cell 1
# <a href="https://colab.research.google.com/github/eklavya0948-max/electric-load-forecasting-weather-time/blob/main/LSTM.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] Cell 2
# # What is LSTM?
# Long short term memory or LSTM is a specialised recurrent neural network (RNN) which excels in capturing long term dependencies which make it good choice to predict sequential data with temporal dependencies.

# %% [markdown] Cell 3
# Importing data set

# %% [code] Cell 4

# %% [markdown] Cell 5
# Importing required libraries

# %% [code] Cell 6
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# %% [markdown] Cell 7
# Loading cleaned data.csv

# %% [code] Cell 8
df = pd.read_csv("cleaned data.csv")

# %% [markdown] Cell 9
# Defining Features

# %% [code] Cell 10
data = df[["temperature","humidity","load"]]

# %% [markdown] Cell 11
# Scaling: To normalise values between 0 and 1 to make training process effecient.

# %% [code] Cell 12
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# %% [markdown] Cell 13
# Creating Sequence (24 hour)

# %% [code] Cell 14
X = []
y = []

time_steps = 24

for i in range(time_steps,len(data_scaled)):
    X.append(data_scaled[i-time_steps:i])
    y.append(data_scaled[i,2])

X = np.array(X)
y = np.array(y)


# %% [markdown] Cell 15
# Train Test Split (80% training 20% testing)

# %% [code] Cell 16
train_size = int(len(X)*0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]


# %% [markdown] Cell 17
# LSTM Model

# %% [code] Cell 18
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)


# %% [markdown] Cell 19
# Training LSTM Model

# %% [code] Cell 20
model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test,y_test)
)

# %% [markdown] Cell 21
# # What are epochs and what 30 epochs and 219 represents?
# 
# 1. If we take textbook definition of epoch in neural network and machine learning then epoch refers to one complete pass through the entire training data set . In simple words it represents how many time same data is fed to neural network.
# 
# For e.g. 1 epoch means model goes through entire training dataset one time.
# 
# 2 epochs means model goes through entire training data set 2 times.
# 
# 2. In our case 30 epochs means our model goes through data set 30 times.
# 
# 3. For each epoch training data is divided into small data chunks called batches. For our data we have batch size= 32 . Since we are using 80% of our total dataset its equal to 8735 *0.8 = 6988 columns . These 6988 columns are divides into batches of 32 columns, the number of batches is calculates ad 6988/32 = 218.375 which is rounded to 219 to ensure all samples are processed. So we have 219 batches with 218 batches of 32 columns and 219th batch with 22 columns.

# %% [markdown] Cell 22
# # What are loss and val_loss?
# 
# 1. Loss refers to training loss . It measures how well a model's predictions match the actual values for the data it has gone through during training. With each epoch model tries to decrease this value by adjustingits weights on features defined. A decreasing training loss means model is learning pattern. In our case loss at start was $10^{-2}$ which was aleready very small , in the end it was also decreased to $10^{-3}$ .
# 
#    i.  It indicates that our model was able to make predictions that were relatively close to the actual values . This suggest that the data is predictable and our features and model framework is suited for thes task .
# 
#    ii. The continious decrase in loss value indicates that our model was consistentely learning and improvising to decrease the overall errors.
# 
# 2. Val_Loss stands for validation loss. It measures how well model's predicition matches actual values which it has not seen during training.
# 
#    In our model initial val_loss was 0.0075 which is decreased to 0.0026 by the 30th epoch. This indicates that our model which is quite good from the start to predict unseen data is not only memorising new data but also learning pattern to predict new unseen data.
# 
# 
# # What are overall conclusions from loss data and val_loss data ?
# 
#  Our loss data is very small and continiously decreasing , same goes with our val_loss data. Combining both results we can conclude that our chosen features and model framework is ideal for our task and since both loss and val_loss values are decreasing our model not only excels in learning training data and patterns but also accurately predict for new unseen data.
# 

# %% [markdown] Cell 23
# Prediction: It is used to store predicitions for evaluation and future forecasting

# %% [code] Cell 24
pred = model.predict(X_test)

# %% [markdown] Cell 25
# Inverse Scaling : Converting prediction to original scale for meaningfull inerpretion.

# %% [code] Cell 26
load_scaler = MinMaxScaler()
load_scaler.fit(df[["load"]])

pred = load_scaler.inverse_transform(pred)
y_test = load_scaler.inverse_transform(y_test.reshape(-1,1))

# %% [markdown] Cell 27
# Evaluation of model using MAE and RMSE

# %% [code] Cell 28
mae = mean_absolute_error(y_test,pred)
rmse = np.sqrt(mean_squared_error(y_test,pred))

print("MAE:",mae)
print("RMSE:",rmse)

# %% [markdown] Cell 29
# # What are your conclusions on model on the basis of MAE and RMSE values?
# Since we lack value of total capacity of substation we will be using maximum observed load to calculate percentage error which is 6306.21 KW.
# 
# 1. MAE relative to max. observed load on substation
# 
#         ( 175.50 / 6306.21) * 100 = 2.78%
# 
# This means model's average prediction error is about 2.78% of the peak load experienced by the system.
# 
# 2. RMSE relative to max. observed load on substation
# 
#         ( 300.23 / 6306.21) * 100 = 4.76%
# 
# This suggests that the larger errors the model makes are typically within 4.76% of the peak load.
# 
# Overall conclusion : Since MAE is 2.76% of overall load and RMSE is 4.76% of overall load our model performs quite good for prediction of complex system of electric grid.

# %% [markdown] Cell 30
# Plotting graph of Actual Load vs Predicted Load

# %% [code] Cell 31
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot(y_test[:200], label="Actual Load")
plt.plot(pred[:200], label="Predicted Load")

plt.legend()
plt.title("Load Prediction using LSTM")

plt.show()
