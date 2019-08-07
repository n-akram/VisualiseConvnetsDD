'''
Function to visualizing-what-convnets-learn.ipynb
date: 26.07.2019

source: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb
Parameters: -m  : For mobile model

'''
import keras
from keras.models import load_model
from keras import models
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import cv2



MOBILE = False
if '-m' in sys.argv:
    MOBILE = True

modelPath = 'sampleModel/'
testSamePath = 'sampleTest/same/'
testNewPath = 'sampleTest/new/'
#sampleImg = 'c1_img_115.jpg' # training set
sampleImg ='c3_img_8.jpg' # test set

def checkShape(model):
    s = []
    i = 1
    for layer in model.layers:
        s = layer.output_shape
    if s[1] is 10:
        return("all")
    else:
        return("mobile")


def loadCorrectmModel():
    global MOBILE
    i = 0
    models = os.listdir(modelPath)
    model = load_model(modelPath + models[i % 2])
    if MOBILE:
        chk = checkShape(model)
        if chk == "all":
            i += 1
            model = load_model(modelPath + models[i % 2])
    else:
        chk = checkShape(model)
        if chk == "mobile":
            i += 1
            model = load_model(modelPath + models[i % 2])
    return(model)

model= loadCorrectmModel()
#model.summary()  # As a reminder.

img = cv2.imread(sampleImg)#, target_size=(150, 150))
resized_image = cv2.resize(img, (150, 150))
norm_image = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) #input was normalized
img_tensor = np.expand_dims(norm_image, axis=0)
#print(img_tensor.shape)

#plt.imshow(img_tensor[0])
#plt.show()

'''
In order to extract the feature maps we want to look at, we will create a Keras model that takes batches of images as input, 
and outputs the activations of all convolution and pooling layers. To do this, we will use the Keras class Model. A Model is 
instantiated using two arguments: an input tensor (or list of input tensors), and an output tensor (or list of output tensors). 
The resulting class is a Keras model, just like the Sequential models that you are familiar with, mapping the specified 
inputs to the specified outputs. What sets the Model class apart is that it allows for models with multiple outputs, unlike 
Sequential.
'''


# Extracts the outputs of the top 6 layers: Because after that we have flatten and dense layers
layer_outputs = [layer.output for layer in model.layers[:6]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

'''
When fed an image input, this model returns the values of the layer activations in the original model. 
'''
# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
#print(first_layer_activation.shape)

#plt.matshow(first_layer_activation[0, :, :, 2], cmap='viridis')
#plt.show()

# These are the names of the layers, so we can have them as part of our plot
layer_names = []
for layer in model.layers[:6]:
    layer_names.append(layer.name)

images_per_row = 8

# Now to display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map (basically the 4th element in shape)
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # tile the activation channels in this matrix
    # Basically creating a big grid using number of images per row and number of columns
    # This grid will be used to plot all individual images
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            # Not clear the need for it
            #channel_image -= channel_image.mean()
            #channel_image /= channel_image.std()
            #channel_image *= 64
            #channel_image += 128
            #channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()