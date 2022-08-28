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

Include your code here

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
