# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Read the csv file and create the Data frame using pandas.

### STEP 2:
Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.
### STEP 3:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train. 
### STEP 4:
Create a model with the desired number of nuerons and one output neuron.
### STEP 5: 
Follow the same steps to create the Test data. But make sure you combine the training data with the test data.
### STEP 6:
Make Predictions and plot the graph with the Actual and Predicted values.
## PROGRAM
```python3
import pandas as pd
import numpy as np
train_data=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/trainset.csv")
train_data.head()
```
```python3
train_data_array=np.array((train_data.iloc[:,1:2]).values)
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train_data_array=sc.fit_transform(train_data_array)
```
```python3
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(train_data_array[i-60:i,0])
  y_train_array.append(train_data_array[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
```
```python3
length = 60
n_features = 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
model=Sequential([
    SimpleRNN(120,input_shape=(length,n_features)),
    Dense(1)
])
model.compile(optimizer="adam",loss='mse')
```
```python3
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=50)
model.fit(X_train1,y_train,epochs=500,batch_size=32,verbose=2,callbacks=[early_stop])
```
```python3
test_data=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/testset.csv")
test_data_array=np.array(test_data.iloc[:,1:2].values)
test_data_array=sc.transform(test_data_array)
X_test_data=[]
X_test_data=list(train_data_array)+list(test_data_array)
X_test=[]
for i in range(60,(len(X_test_data))):
  X_test.append(X_test_data[i-60:i])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
```
```python3
predicted_stock_price=model.predict(X_test)
predicted_stock_price=MinMaxScaler().fit_transform(predicted_stock_price)
```
```python3
import matplotlib.pyplot as plt
plt.plot(np.arange(0,1384),X_test_data,color='red',label="Test(Real) Google Stock Price")
plt.plot(np.arange(60,1384),predicted_stock_price,color='blue',label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.show()
```

## OUTPUT

### True Stock Price, Predicted Stock Price vs time

![image](https://user-images.githubusercontent.com/65499285/194717348-80b01054-1f67-4861-9e91-f1b32dc62825.png)

### Mean Square Error

Include the mean square error

## RESULT
Hence, we have successfully created a Simple RNN model for Stock Price Prediction.