# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
1 Input Layer<br>
4 hidden layers - 64 neurons, 32 neurons, 16 neurons, 8 neurons<br>
1 Output layer - 4 neurons<br>

![nn](https://user-images.githubusercontent.com/65499285/189124834-1efbdb4c-79ac-48b2-bcbb-b1d13ff640b0.svg)

## DESIGN STEPS

### STEP 1:
We start by reading the dataset using pandas.
### STEP 2:
The dataset is then preprocessed, i.e, we remove the features that don't contribute towards the result.
### STEP 3:
The null values are removed aswell
### STEP 4:
The resulting data values are then encoded. We, ensure that all the features are of the type int, or float, for the model to better process the dataset.
### STEP 5:
Once the preprocessing is done, we split the avaliable data into Training and Validation datasets.
### STEP 6:
The Sequential model is then build using 4 dense layers(hidden) and, 1 input and output layer.
### STEP 7:
The model is then complied and trained with the data. A call back method is also implemented to prevent the model from overfitting.
### STEP 8:
Once the model is done training, we validate and use the model to predict values.
## PROGRAM
```python3
import pandas as pd
data = pd.read_csv("customers.csv")
data.head()
```
```python3
data_cleaned=data.drop(columns=["ID","Var_1"])
data_col=list(data_cleaned.columns)
print("The shape of the data before removing null values is\nRow:"+str(data_cleaned.shape[0])+"\nColumns:"+str(data_cleaned.shape[1]))
```

```python3
pd.DataFrame(data_cleaned.isnull().sum())
data_cleaned=data_cleaned.dropna(axis=0)
print("The shape of the data after removing null values is\nRow:"+str(data_cleaned.shape[0])+"\nColumns:"+str(data_cleaned.shape[1]))

```

```python3
data_col_obj=list()
for c in data_col:
  if data_cleaned[c].dtype=='O':
      data_col_obj.append(c)
data_col_obj.remove("Segmentation")
print("The Columns/Features that have Objects(dataType) before encoding are:\n")
print(data_col_obj)

from sklearn.preprocessing import OrdinalEncoder
data_cleaned[data_col_obj]=OrdinalEncoder().fit_transform(data_cleaned[data_col_obj])
from sklearn.preprocessing import MinMaxScaler
data_cleaned[["Age"]]=MinMaxScaler().fit_transform(data_cleaned[["Age"]])
data_cleaned.head()

from sklearn.preprocessing import OneHotEncoder
y=data_cleaned[["Segmentation"]].values
y=OneHotEncoder().fit_transform(y).toarray()
pd.DataFrame(y)
```

```python3
X=data_cleaned.iloc[:,:-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model=Sequential([
    Dense(64,input_shape=X_train.iloc[0].shape,activation="relu"),
    Dense(32,activation='tanh'),
    Dense(16,activation='relu'),
    Dense(8,activation='tanh'),
    Dense(4,activation='softmax'),
])

model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=15)
model.fit(x=X_train,y=y_train,
          epochs=400,
          validation_data=(X_test,y_test),
          verbose=0, 
          callbacks=[early_stop]
          )
```


```python3
metrics = pd.DataFrame(model.history.history)
metrics.iloc[metrics.shape[0]-1,:]
```

```python3
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(metrics[['accuracy','val_accuracy']])
plt.legend(["Training Accuracy","Validation Accuracy"])
plt.title("Accuracy vs Test Accuracy")
plt.subplot(1,2,2)
plt.plot(metrics[['loss','val_loss']])
plt.legend(["Training Loss","Validation Loss"])
plt.title("Loss vs Test Loss")
plt.show()
```

```python3
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
predictions=np.argmax(model.predict(X_test),axis=1)
y_test=np.argmax(y_test, axis=1)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
import seaborn as sn
sn.heatmap(confusion_matrix(y_test,predictions))
plt.show()
```
## Dataset Information
![image](https://user-images.githubusercontent.com/65499285/188649793-d5045c93-f721-47fd-b504-021494fcb256.png)


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/65499285/188649855-6e824cbd-3148-424e-bc24-deb0e093d2a8.png)


### Classification Report
![image](https://user-images.githubusercontent.com/65499285/188649904-60a32234-a508-41db-b968-039259a9a2d1.png)

### Confusion Matrix
![image](https://user-images.githubusercontent.com/65499285/188650140-be55fbbd-6ae9-4d37-9ccf-cede70c45fc1.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/65499285/188653473-b4378d95-9d21-427a-93ed-47af07b09063.png)

## RESULT
Hence we have constructed a Neural Network model for Multiclass Classification.
