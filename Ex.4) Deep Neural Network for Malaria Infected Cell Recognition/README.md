# Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
We have a directory with a number of images of infected cells and uninfected cells. Using an image generator, we augment the data into multiple images. We pass the data to the model and train the model accordingly using the required number of neurons. 

## Neural Network Model

![Neural Networks](https://user-images.githubusercontent.com/65499285/193041366-f95ff404-8ad6-497c-be7f-dc87c2e7c3a1.svg)

## DESIGN STEPS

### STEP 1:
Define the directory for the dataset. Extract the dataset files if needed.
### STEP 2:
Define the image Generator engine with the necessary parameters.
### STEP 3:
Pass the directory to the image generator.
### STEP 4:
Define the model with appropriate neurons.
### STEP 5:
Pass the training and validation data to the model.
### STEP 6:
Plot the necessary graphs. 

## PROGRAM
```python3
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
```
```python3
my_data_dir = 'cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
```
```python3
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
w=max(set(dim1),key=dim1.count)
h=max(set(dim2),key=dim2.count)
image_shape = (w,h,3)
```
```python3
sns.jointplot(x=dim1,y=dim2)
```
```python3
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
```
```python3
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense
model=Sequential()
model.add(layers.Input(shape= image_shape))
model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
model.add(layers.MaxPool2D())
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics="accuracy")
```
```python3
batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
```
```python3
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
```
```python3
results = model.fit(train_image_gen,epochs=10,
                              validation_data=test_image_gen
                             )
```
```python3
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
model.metrics_names
model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
predictions = pred_probabilities > 0.5
print(classification_report(test_image_gen.classes,predictions))
confusion_matrix(test_image_gen.classes,predictions)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![output](https://user-images.githubusercontent.com/65499285/193079633-c4546b4d-bfd5-43d3-a79d-7603a2fb814b.png)

### Classification Report

![image](https://user-images.githubusercontent.com/65499285/193079690-361c903a-c3a0-4a66-8df7-939d6935df7f.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/65499285/193080137-fb73fe4e-88bd-4381-b491-3ff649715a5d.png)

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/65499285/193200885-6156cf87-1cb7-48c9-9f50-7e7e73f02e77.png)

## RESULT
Hence we have successfully created a deep neural network for Malaria infected cell recognition and analyzed the performance.
