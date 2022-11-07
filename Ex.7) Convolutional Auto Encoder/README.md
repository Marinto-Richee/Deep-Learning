# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
We use the mnish dataset for this implementation of Convolutional Auto Encoder. We ignore the classification of each image, as we aim to denoise rather than classify.
<br>We load the images, and add noises to the data. The noised images will provide to be the input, and the orignial untouched images will be passed as the output. 
Same goes with the test/validation data aswell.
<br>The model is built using Conv2D, MaxPooling2D, and UpSampling2D layers. We are not using dense layers for this implementation. 
The rest of the flow, remains the same as we do for any other Deep Neural Implementataion.

## Convolution Autoencoder Network Model

![nn](https://user-images.githubusercontent.com/65499285/200237286-29f2220e-c59c-4951-bf6f-3b353fd420ac.svg)

## DESIGN STEPS

### STEP 1:
Import the necessary libraries and dataset.

### STEP 2:
Load the dataset and scale the values for easier computation.

### STEP 3:
Add noise to the images randomly for both the train and test sets.

### STEP 4:
Build the Neural Model using Convolutional, Pooling and Up Sampling layers. Make sure the input shape and output shape of the model are identical.

### STEP 5:
Pass test data for validating manually. 

### STEP 6: 
Plot the predictions for visualization.

## PROGRAM

```python3
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import  matplotlib.pyplot as plt
import random
```
```python3
(x_train,_),(x_test,_)=mnist.load_data()
x_train=x_train/255.#float
x_test=x_test/255.
x_train=np.reshape(x_train,(-1,28,28,1))
x_test=np.reshape(x_test,(-1,28,28,1))
```
```python3
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
```
```python3
model=Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(16,(5,5),activation='relu'),
    layers.MaxPool2D((2,2),padding='same'),
    layers.Conv2D(8,(3,3),activation='relu'),
    layers.MaxPool2D((2,2),padding='same'),
    layers.Conv2D(8,(3,3),activation="relu",padding='same'),
    layers.UpSampling2D((2,2)),
    layers.Conv2D(16,(5,5),activation='relu',padding='same'),
    layers.UpSampling2D((3,3)),
    layers.Conv2D(1,(3,3),activation='sigmoid')
])
model.summary()
```
```python3
 model.compile(optimizer='adam', loss='binary_crossentropy')
 model.fit(x_train_noisy,x_train,epochs=5,batch_size=64)
```
```python3
n1 = random.choices(range(len(x_test)),k=10)
predict_img=model.predict(x_test_noisy[n1])
plt.figure(figsize=(20, 5))
for i in range(1, n + 1):
    # Display original
    plt.subplot(3, n, i)
    plt.axis("off")
    plt.imshow(x_test[n1[i-1]].reshape(28, 28),"gray")
    # Display noisy
    plt.subplot(3, n, i+n)
    plt.axis("off")
    plt.imshow(x_test_noisy[n1[i-1]].reshape(28, 28))
    # Display reconstruction
    plt.subplot(3, n, i + 2*n)
    plt.axis("off")
    plt.imshow(predict_img[i-1].reshape(28, 28))    
plt.show()
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/65499285/200269971-62f29cd3-818b-4eb4-8e88-2ca3226a3b1e.png)


### Original vs Noisy Vs Reconstructed Image

![image](https://user-images.githubusercontent.com/65499285/200238208-81b7ba36-86b2-46f5-9d33-604ae7437534.png)

## RESULT
Hence, we have sucessfully implemented a Convolutional Auto Encoder for Denoising.
