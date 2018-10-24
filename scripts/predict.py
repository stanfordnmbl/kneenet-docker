import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
from torch.utils.data import sampler
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from torch.autograd import Variable
import sklearn
from sklearn.metrics import confusion_matrix
import scipy.ndimage
import cv2
import math
import dill
import pickle
import torchvision.models as models

def default_preprocessing(image, 
                          min_hw_ratio = 1, 
                          output_width = 299, 
                          output_height = 299):

    # Trim equal rows from top and bottom to get a square image
    r, c = image.shape
    image_hw_ratio = r/c
    r_to_keep = c * min_hw_ratio
    r_to_delete = r - r_to_keep
    remove_from_top = int(math.ceil(r_to_delete/2))
    remove_from_bottom = int(math.floor(r_to_delete/2))
    image_top_bottom_trimmed = image[remove_from_top:(r-remove_from_bottom),:]

    # resample to get the desired image size
    image_resampled = cv2.resize(image_top_bottom_trimmed, dsize=(output_width, output_height), interpolation=cv2.INTER_CUBIC)
    
    # Normalize pixel values to take the range [0,1]
    image_clean = image_resampled - np.mean(image_resampled)
    image_clean = image_clean / np.std(image_clean)
    image_clean = ((image_clean - np.min(image_clean)) / (np.max(image_clean) - np.min(image_clean)))  

    # Stack into three channels
    image_clean_stacked = np.dstack((image_clean, image_clean, image_clean))
    image_clean_stacked = np.moveaxis(image_clean_stacked, -1, 0)  
    
    # Implement ImageNet Standardization
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    imagenet_std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
    image_clean_stacked = (image_clean_stacked - imagenet_mean) / imagenet_std
    
    return image_clean_stacked


# Initialize a model
pretrained_model = torchvision.models.densenet169(pretrained = False)
#pretrained_model_test = torchvision.models.densenet169(pretrained = True)
#pretrained_model_test = pretrained_model_test.double()

for param in pretrained_model.parameters():
    param.requires_grad = False
pretrained_model.aux_logits = False

# Modify the output layer of the densenet to fit our number of output classses
num_features = pretrained_model.classifier.in_features
num_features_knee = 14976

out_features = pretrained_model.classifier.out_features
pretrained_model.classifier = nn.Linear(num_features_knee, 5)

for param in pretrained_model.classifier.parameters():
    param.requires_grad = False

# Load the model's trained weights
pretrained_model.load_state_dict(torch.load('./models/KneeNet.0', 
                                           map_location=lambda storage, 
                                           loc: storage))
pretrained_model.train(False)
pretrained_model = pretrained_model.float()


output_file = open('output/predictions.csv', 'w')
output_file.write('filename,prediction\n')

for img in glob.glob('input/*'):
    print("Processing file %s..." % img)
    sample_input = cv2.imread(img,0).astype('float') 
    sample_input = default_preprocessing(sample_input)

    sample_input = sample_input.reshape((1,) + sample_input.shape)
    sample_input = torch.from_numpy(sample_input)
    sample_input = sample_input.float()

    img_size = 299
    input_np = np.random.uniform(0, 1, (1, 3, img_size, img_size))
    input_dummy = Variable(torch.FloatTensor(input_np))

    probs = pretrained_model(sample_input)[0]

    print(probs)
    output_file.write('%s,%d\n' % (img,int(np.argmax(probs))))

output_file.close()
print("Processing finished.")
