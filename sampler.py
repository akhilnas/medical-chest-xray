
import os
import glob

# General Data Manipulation Libraries
import numpy as np
import pandas as pd

# Image Manipulation
from cv2 import cv2

# PyTorch Libraries
import torch
from torch.functional import Tensor
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import torch.optim as optim

from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# Plotting Libraries
import matplotlib.pyplot as plt

# Import Model Structure
from model import vgg16_tuned

def image_transforms(img):
    '''
    Performs the Necessary Image Transformations to the image for feeding into
    model for evaluation
    img:  Image in Numpy array ,Shape --> [H, W, C]
    '''
    # Resizing Image Dimensions
    img = cv2.resize(img, (224,224))

    # Normalization of Image as per mean and std values given    
    # Channel 1
    img[:,:,0] = (img[:,:,0] -0.485) / 0.229
    # Channel 2
    img[:,:,1] = (img[:,:,1] -0.456) / 0.224
    # Channel 3
    img[:,:,2] = (img[:,:,2] -0.406) / 0.225    

    # Readjust indices for Passing to model [H,W,C] to [batch_size=1, C, H, W]  
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    # Converting Numpy array to Tensor
    img = Tensor(img)
    return img


if __name__ == '__main__':

    # CUDA Check
    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'  
        torch.set_default_tensor_type('torch.FloatTensor')

    BASE_DIR = os.getcwd()

    # Print PyTorch version
    print(f'PyTorch Version {torch.__version__}')
    print(f'Torchvision Version {torchvision.__version__}')    

    # Load Data
    ## Find all files in data subfolder
    for name in glob.glob(BASE_DIR + '/data/*'):
        img = cv2.imread(name)    
    ## Perform Necessary Transformations
    img = image_transforms(img)
    ## Transfer data to Device
    img = img.to(device)

    # Load the Model 
    model = vgg16_tuned()
    model.load_state_dict(torch.load('state_dict_model.pt'), strict=False)    
    model.eval()
    model.to(device)

    # Evaluate Image
    output = model(img)
    _, preds = torch.max(output, 1)

    print(output)
    class_name = 'NORMAL' if preds == 0 else 'PNEUMONIA'
    print(f'The Output Class is {class_name}.')

def image_classifier():

    # CUDA Check
    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = 'cpu'  
        torch.set_default_tensor_type('torch.FloatTensor')

    BASE_DIR = os.getcwd()
    # Load Data
    ## Find all files in data subfolder
    for name in glob.glob(BASE_DIR + '/static/uploads/*'):
        img = cv2.imread(name)    
    ## Perform Necessary Transformations
    img = image_transforms(img)
    ## Transfer data to Device
    img = img.to(device)

    # Load the Model 
    model = vgg16_tuned()
    model.load_state_dict(torch.load('state_dict_model.pt'), strict=False)
    model.eval()
    model.to(device)

    # Evaluate Image
    output = model(img)
    _, preds = torch.max(output, 1)

    print(output)
    class_name = 'NORMAL' if preds == 0 else 'PNEUMONIA'

    return class_name




