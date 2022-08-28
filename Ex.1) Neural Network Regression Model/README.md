# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for a simple dataset with one input and one output.

## THEORY

We create a simple dataset with one input and one output.This data is then divided into test and training sets for our Neural Network Model to train and test on. <br>
The NN Model contains 5 nodes in the first layer, 10 nodes in the following layer, which is then connected to the final output layer with one node/neuron.
The Model is then compiled with an loss function and Optimizer, here we use MSE and rmsprop. <br>The model is then train for 2000 epochs.<br> We then perform an evaluation of the model with the test data. An user input is then predicted with the model. Finally, we plot the Error VS Iteration graph for the given model.

## Neural Network Model

![Neural Model](https://user-images.githubusercontent.com/65499285/187072101-d6415f29-385b-46aa-b4b9-131d557ca630.svg)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
x=[]
y=[]
for i in range(60):
  num = i+1
  x.append(num)
  y.append(num*10+1) 
df=pd.DataFrame({'Input': x, 'Output': y})
df.head()

inp=df[["Input"]].values
out=df[["Output"]].values
Input_train,Input_test,Output_train,Output_test=train_test_split(inp,out,test_size=0.33)
Scaler=MinMaxScaler()
Scaler.fit(Input_train)
Scaler.fit(Input_test)
Input_train=Scaler.transform(Input_train)
Input_test=Scaler.transform(Input_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model=Sequential([Dense(5,activation='relu'),
                  Dense(10,activation='relu'),
                  Dense(1)])
model.compile(loss="mse",optimizer="rmsprop")
history=model.fit(Input_train,Output_train, epochs=4000,batch_size=32)

prediction_test=int(input("Enter the value to predict:"))
preds=model.predict(Scaler.transform([[prediction_test]]))
print("The prediction for the given input "+str(prediction_test)+" is:"+str(int(np.round(preds))))

model.evaluate(Input_test,Output_test)

import matplotlib.pyplot as plt
plt.suptitle("   Marinto Richee")
plt.title("Error VS Iteration")
plt.ylabel('MSE')
plt.xlabel('Iteration')
plt.plot(pd.DataFrame(history.history))
plt.legend(['train'] )
plt.show()
```

## Dataset Information

![image](https://user-images.githubusercontent.com/65499285/187072539-fe596d8e-ec7b-46f6-8cf3-61f3deed4378.png)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/65499285/187072875-27433ea7-c9bc-43a7-8017-59b84b7d6389.png)

### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/65499285/187072948-40f531fc-61b0-45f7-8877-fe5b874205c8.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/65499285/187072962-ff4d3337-3e0b-48e5-9c34-6a40728f1435.png)

## RESULT
Hence, a simple Neural Network Model has been implemented successfully.
