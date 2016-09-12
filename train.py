'''
Created on Sep 12, 2016

@author: liudanny
'''

from core.imif_digits import *
import cv2
import sys

# Initializes our image identifier class
imif = imif_digits()

# Trains and saves model
# imid.train_and_save_model('data/MNIST_digits', '../trained_models/mnist_digits.ckpt')

# Loads saved model
imif.train_and_save_model('data/MNIST_digits', 'trained_models/mnist_digits.ckpt')
