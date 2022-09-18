# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

![nn](https://user-images.githubusercontent.com/65499285/190903314-151ecc46-ab8a-47e9-8c9e-af2d699c38c1.svg)

![model](https://user-images.githubusercontent.com/65499285/190903321-249a6aae-bc42-4d97-9f69-b488f12215a2.png)


## DESIGN STEPS

### STEP 1:
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

### STEP 2:
Then we move to normalization and encoding of the data.

### STEP 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

### STEP 4:
Early Stopping is defined so that the model doesn't overfit itself. We then train the model with the training data.

### STEP 5:
The necessary Validating parameters are visualized for inspection.

### STEP 6:
Finally, we pass handwritten digits to the model for prediction.

## PROGRAM
```python3
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense
(X_train,y_train),(X_test,y_test)=mnist.load_data()

```
```python3
import matplotlib.pyplot as plt
import random
num=random.randint(0,X_train.shape[0])
plt.axis("off")
plt.imshow(X_train[num],cmap="gray")
plt.show()
```
```python3
print("Highest Value before normalization: "+str(X_test.max()))
X_test_scaled=X_test/255.0
X_train_scaled=X_train/255.0
print("Highest Value after normalization: "+str(X_test_scaled.max()))
```
```python3
X_test_scaled=X_test_scaled.reshape(-1,28,28,1)
X_train_scaled=X_train_scaled.reshape(-1,28,28,1)
from sklearn.preprocessing import LabelBinarizer
y_test_1_encode = LabelBinarizer().fit_transform(y_test)
y_train_1_encode = LabelBinarizer().fit_transform(y_train)
test_num=random.randint(0,y_test.shape[0])
print("The value before One-Hot Encoding is "+str(y_test[test_num])+"\nArfter applying One-Hot Encoding is "+str(y_test_1_encode[test_num]))
```
```python3
model=Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(Dense(16,activation='tanh'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics="accuracy")
```
```python3
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=0.5)
model.fit(X_train_scaled,y_train_1_encode,128,10,validation_data=(X_test_scaled,y_test_1_encode),callbacks=[early_stop],verbose=2)
```
```python3
import matplotlib.pyplot as plt
import pandas as pd
metrics=pd.DataFrame(model.history.history)
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
from sklearn.metrics import confusion_matrix,classification_report
from seaborn import heatmap
x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
y_test_1_encode_=np.argmax(y_test_1_encode,axis=1)
display(pd.DataFrame(confusion_matrix(y_test_1_encode_,x_test_predictions)))
heatmap(pd.DataFrame(confusion_matrix(y_test_1_encode_,x_test_predictions)))
print(classification_report(y_test_1_encode_,x_test_predictions))

```
```python3
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
img_2_ = image.load_img('/content/2.png')
img_3_ = image.load_img('/content/3.png')
img_5_ = image.load_img('/content/5.png')
lst_img_=[img_2_,img_3_,img_5_]
for j in lst_img_:
  img  = tf.convert_to_tensor(np.asarray(j))
  img = tf.image.resize(img,(28,28))
  img = tf.image.rgb_to_grayscale(img)
  img = img.numpy()/255.0
  plt.axis("off")
  plt.imshow(img.reshape(28,28),cmap="gray")
  x_single_prediction = np.argmax(
      model.predict(img.reshape(1,28,28,1)),
      axis=1)
  plt.title("The Prediction is "+str(x_single_prediction))
  plt.show()
  print("\n")
```
```python3
import cv2
kernel=np.ones((20,20),np.uint8)
for j in lst_img_:
  img  = tf.convert_to_tensor(np.asarray(j))
  img =cv2.dilate(np.array(img),kernel)
  img = tf.image.resize(img,(28,28))
  img = tf.image.rgb_to_grayscale(img)
  img = img.numpy()/255.0
  plt.axis("off")
  plt.imshow(img.reshape(28,28),cmap="gray")
  x_single_prediction = np.argmax(
      model.predict(img.reshape(1,28,28,1)),
      axis=1)
  plt.title("The Prediction is "+str(x_single_prediction))
  plt.show()
  print("\n")

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/65499285/190903441-83cff0f2-493d-4cbd-a4ca-778a292787d5.png)

### Classification Report

![image](https://user-images.githubusercontent.com/65499285/190903861-f047e613-b37d-4c5d-bda0-58ff82cad331.png)

### Confusion Matrix

![image](https://user-images.githubusercontent.com/65499285/190903464-efdc95b9-bde7-408a-af67-707d88cd0efe.png)

### New Sample Data Prediction

#### Input:

![2](https://user-images.githubusercontent.com/65499285/190903873-e97aa66a-0a4c-4fef-afd6-2bc76f448baa.png)

![5](https://user-images.githubusercontent.com/65499285/190903876-b14641e0-5c07-48ff-8f0c-f7d094113a14.png)

![3](https://user-images.githubusercontent.com/65499285/190903878-d7aab851-caca-4ffb-b744-86c5cfe54a65.png)

#### Output:
##### Without Dilation:

![image](https://user-images.githubusercontent.com/65499285/190903931-208ec440-5d19-4892-8506-82f49a19dcdd.png)

##### With Dilation:

![image](https://user-images.githubusercontent.com/65499285/190903949-530d877f-01e0-4989-8a8d-5c67638dd4ee.png)

## RESULT
Hence, a Convolutional Neural Model has been built to predict the handwritten digits.
