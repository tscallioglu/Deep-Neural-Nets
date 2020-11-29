import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_functions import *


plt.rcParams['figure.figsize'] = (5.0, 4.0) # setting default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'




train_dataset = h5py.File('train_catvnoncat.h5', "r")
train_x_orig = np.array(train_dataset["train_set_x"][:]) #  train set features
train_y= np.array(train_dataset["train_set_y"][:]) # train set labels

test_dataset = h5py.File('test_catvnoncat.h5', "r")
test_x_orig = np.array(test_dataset["test_set_x"][:]) # test set features
test_y= np.array(test_dataset["test_set_y"][:]) # test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))
    


# Exploring dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))



# Reshaping the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardizing data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("\ntrain_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape)+"\n")


### CONSTANTS ###
layers_dims = [12288, 20, 10, 5, 1] #  4-layer model - Nodes in every layer




parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1000, print_cost = True)

print("\nTrain")
predictions_train = predict(train_x, train_y, parameters)

print("\nTest")
predictions_test = predict(test_x, test_y, parameters)

print("\nTrain Image Indexs Mislabelled by This Program:")
print_mislabeled_images(classes, train_x, train_y, predictions_train)

print("\nTest Image Indexs Mislabelled by This Program:")
print_mislabeled_images(classes, test_x, test_y, predictions_test)