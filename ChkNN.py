#ChkNN.py
#
#januar 2022
#
#Alenka Trpin
# here used cervical cancer dataset, accessible on https://www.kaggle.com/datasets/ofriharel/224-224-cervical-cancer-screening


from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision 

import glob 
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import NN as hypnn

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, metrics 
from sklearn.model_selection import cross_val_score

import torch 
import torchvision 
import glob 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import neighbors, metrics 
from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import MinMaxScaler  
from sklearn import preprocessing

from pylmnn import LargeMarginNearestNeighbor as LMNN

from sklearn.metrics import fbeta_score, make_scorer

from torch.utils.data import ConcatDataset

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, KFold, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from pylmnn import LargeMarginNearestNeighbor as LMNN


data_dir = 'CervicalC/train'

list_imgsC = glob.glob(data_dir + "/**/*.jpg")
#list_imgs1 = glob.glob(data_dir + "/*.jpg")
#list_img_classes = glob.glob("/**/*.jpg")
print(f"There are {len(list_imgsC)} images in the dataset {data_dir}")

#create dataloader with required transforms 

tc = transforms.Compose([
	transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()              

    ])

image_datasetsC = datasets.ImageFolder(data_dir, transform=tc)
dloader = torch.utils.data.DataLoader(image_datasetsL, batch_size=10, shuffle=False)
print(len(image_datasetsL))


#define a CNN embeddings 
######################### This part is from paper Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2020). Hyperbolic image embeddings. In Proceedings of the IEEE/CVF ######################### conference on computer vision and pattern recognition (pp. 6418-6428).
######################## and it can be found at: https://github.com/leymir/hyperbolic-image-embeddings/blob/master/README.md

class Net(nn.Module):
	def __init__(self, args):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 200, 5, 1)   
		self.conv2 = nn.Conv2d(200, 500, 5, 1) 
		self.fc1 = nn.Linear(500, 200) 
		self.fc2 = nn.Linear(24500, 500)		
		self.tp = hypnn.ToPoincare(c=-1, train_x=100, train_c=100, ball_dim=1000)
		self.mlr = hypnn.HyperbolicMLR(ball_dim=1000, n_classes=3, c=-1)  
		self.dropout = nn.Dropout(0.5)         # Apply dropout

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)  
		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)
		x = x.view(x.size(0), -1)
		x = F.relu(self.fc2(x))
		x = self.dropout(x)        # Apply dropout
		x = self.tp(x)
		return F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1)
########################
########################
########################

#evaluate the model
modelblt = Net(100)
modelblt.eval()


print(len(dloader))
#print(dloader)


# Select the desired layer-------to get attributes from last layer in CNN model------------------------------
layer = modelblt._modules.get('dropout')
outputs1 = []


# Select the desired layer-------to get attributes from last layer in CNN model------------------------------
#copy features from penultimate layer (model, input, output)
def copy_features(m, i, o):
    o = o[:, :].detach().numpy().tolist()  
    outputs1.append(o)

#Define and attach hook to the penulimate layer
_ = layer.register_forward_hook(copy_features)


# Generate image's features for all images in dloader and saves them in the list outputs

for X, y in dloader:
    _ = modelblt(X)
print(len(outputs1))


# flatten list of features to remove batches
list_features1 = [item for sublist in outputs1 for item in sublist]

print(len(list_features1))    #vseh features je 643
print(np.array(list_features1[0]).shape)   

Xmy = np.array(list_features1)
print(f"Mn X {Xmy.shape}")
#print(Xa1[0])


filenameC = os.listdir('CervicalC/train/')

yC = []
for i in range(1, len(list_imgsC)+1): #len(list_embeddings)
    sample_idx = i#torch.randint(len(test_dataset), size=(1,)).item()
    img, label = image_datasetsC[sample_idx-1]
    yC.append(filenameC[label])   #extend()
#print(zz)
#print(f"[{', '.join(yy1)}]")
print(len(yC))



def poincare_distance(row1, row2):
    #print('len(row1) = ',len(row1))
    #print('len(row1) = ',len(row1))
    distance = 0.0
    d = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    d = (row1 - row2)*(row1 - row2)
    d = d.sum()
    
    norm1 = row1 * row1
    norm1 = norm1.sum()
    
    norm2 = row2 * row2
    norm2 = norm2.sum()
    
    m = 1 + 2*d/((1-norm1)*(1-norm2))
    
    distance = np.log(m + np.sqrt(m*m - 1))
    #distance = math.log(m + np.sqrt(m*m - 1))
    
    return(distance)



mmscaler = preprocessing.MinMaxScaler()    
Xminmaxl1 = mmscaler.fit_transform(Xmy)
#print(Xminmaxb1)

"""
print("Euclidean kNN:")
k_scoresEMc = []
k_scoresEMstdc = []
for k in range(1,11):
    #knn = neighbors.KNeighborsClassifier(n_neighbors=k, leaf_size=3, metric='pyfunc', metric_params={"func":poincare_distance}, n_jobs=3)	
    #for testing
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=3)    
    #fit the model
    knn.fit(Xmy,yC)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knn, Xmy, yC, cv=10, scoring='accuracy')
    k_scoresEMc.append(scores.mean())
    k_scoresEMstdc.append(scores.std())
print(k_scoresEMc)
print(k_scoresEMstdc)


"""

def poincare_distance(row1, row2):
    #print('len(row1) = ',len(row1))
    #print('len(row1) = ',len(row1))
    distance = 0.0
    d = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    d = (row1 - row2)*(row1 - row2)
    d = d.sum()
    
    norm1 = row1 * row1
    norm1 = norm1.sum()
    
    norm2 = row2 * row2
    norm2 = norm2.sum()
    
    m = 1 + 2*d/((1-norm1)*(1-norm2))
    
    distance = np.log(m + np.sqrt(m*m - 1))
    #distance = math.log(m + np.sqrt(m*m - 1))
    
    return(distance)



mmscaler = preprocessing.MinMaxScaler()
Xminmaxc = mmscaler.fit_transform(Xmy)
#print(Xminmaxb1)

"""
print("PM kNN, without LMNN:")

#brezLMNN
k_scoresPMc = []
k_scoresPMstdc = []
for k in range(1,11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, leaf_size=3, metric='pyfunc', metric_params={"func":poincare_distance}, n_jobs=3)
    #for testing
    #knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='euclidean')    #algorithm='ball_tree' - for a custom metric
    #fit the model
    knn.fit(Xmy,yC)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knn, Xmy,yC, cv=10, scoring='accuracy')
    k_scoresPMc.append(scores.mean())
    k_scoresPMstdc.append(scores.std())
print(k_scoresPMc)
print(k_scoresPMstdc)

"""


k_train, k_test, max_iter = 3, 3, 180	

# Instantiate the metric learner
lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter)

# Train the metric learner
lmnn.fit(Xminmaxc,yC)

print("PM kNN, with LMNN")
k_scoresPMLc = []
k_scoresPMLstdc = []
for k in range(1,11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, leaf_size=3, metric='pyfunc', metric_params={"func":poincare_distance}, n_jobs=3)	
    #for testing
    #knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='euclidean')    #algorithm='ball_tree' - for a custom metric
    #fit the model
    knn.fit(lmnn.transform((Xminmaxc)),yC)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knn, lmnn.transform((Xminmaxc)), yC, cv=10, scoring='accuracy')
    k_scoresPMLc.append(scores.mean())
    k_scoresPMLstdc.append(scores.std())
print(k_scoresPMLc)
print(k_scoresPMLstdc)





