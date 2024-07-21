#rmycnnpPMskin.py
#
#januar 2023
#
#Alenka Trpin

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



data_dir = 'CervicalC/train'

list_imgsL = glob.glob(data_dir + "/**/*.jpg")
#list_imgs1 = glob.glob(data_dir + "/*.jpg")
#list_img_classes = glob.glob("/**/*.jpg")
print(f"There are {len(list_imgsL)} images in the dataset {data_dir}")

#create dataloader with required transforms 

tc = transforms.Compose([
	transforms.Grayscale(),
        transforms.Resize((256, 256)),

        transforms.ToTensor()              

    ])

image_datasetsL = datasets.ImageFolder(data_dir, transform=tc)
dloader = torch.utils.data.DataLoader(image_datasetsL, batch_size=10, shuffle=False)
print(len(image_datasetsL))


#define a CNN embeddings 
######################### This part is from paper Khrulkov, V., Mirvakhabova, L., Ustinova, E., Oseledets, I., & Lempitsky, V. (2020). Hyperbolic image embeddings. In Proceedings of the IEEE/CVF ######################### conference on computer vision and pattern recognition (pp. 6418-6428).
######################## and it can be found at: https://github.com/leymir/hyperbolic-image-embeddings/blob/master/README.md

class Net(nn.Module):
	def __init__(self, args):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 200, 5, 4)   
		self.conv2 = nn.Conv2d(200, 500, 3, 2) 
		self.conv3 = nn.Conv2d(500, 500, 2, 2)
	
		self.fc1 = nn.Linear(500, 200) 

		self.fc2 = nn.Linear(24500, 500)

		self.fc3 = nn.Linear(500, 1000)

		
		self.tp = hypnn.ToPoincare(c=-1, train_x=100, train_c=100, ball_dim=1000)

		
		self.mlr = hypnn.HyperbolicMLR(ball_dim=1000, n_classes=2, c=-1)   

		self.dropout = nn.Dropout(0.5)         # Apply dropout

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x, 2, 2)  

		x = F.relu(self.conv2(x))
		x = F.max_pool2d(x, 2, 2)

		x = x.view(x.size(0), -1)
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.dropout(x)        # Apply dropout
		x = self.tp(x)
		return F.log_softmax(self.mlr(x, c=self.tp.c), dim=-1)
		#return(x)
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


# flatten list of features to remove batches!!!!!!!!!!!!!!----in numpy array....
list_features1 = [item for sublist in outputs1 for item in sublist]

print(len(list_features1))    #vseh features je 643
print(np.array(list_features1[0]).shape)   #dolžina ene feature je 512

Xmy = np.array(list_features1)
print(f"Mn X {Xmy.shape}")
#print(Xa1[0])





filenameL = os.listdir('CervicalC/train/')
#print(filenameL)

yL = []
for i in range(1, len(list_imgsL)+1): #len(list_embeddings)
    sample_idx = i#torch.randint(len(test_dataset), size=(1,)).item()
    img, label = image_datasetsL[sample_idx-1]
    yL.append(filenameL[label])   #extend()
#print(zz)
#print(f"[{', '.join(yy1)}]")
print(len(yL))


"""
print("kNN, basic:")
meanA1my = []
meanP1my = []
meanR1my = []
meanF1my = []

k_scoresEMlc = []
k_scoresEMstdlcresnet = []
for k in range(1,11):
    
    #for testing
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='euclidean', n_jobs=3)    #algorithm='ball_tree' - for a custom metric
    #fit the model
    knn.fit(Xmy,yL)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knn, Xmy, yL, cv=10, scoring='accuracy')
    k_scoresEMlc.append(scores.mean())
    k_scoresEMstdlcresnet.append(scores.std())
    #for UNBALANCED DATASET
    scoringA = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score, average='macro'),
           'recall' : make_scorer(recall_score, average='macro'), 
           'f1_score' : make_scorer(f1_score, average='macro')}

    kfold = StratifiedKFold(n_splits=10) #if it is: kfold = KFold(n_splits=10, random_state=42)-->error: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
    #results = cross_val_score(estimator=kNNa, X=Xa1,y=yL,cv=kfold,scoring=scoring)
    resultsAmy = cross_validate(estimator=knn, X=Xmy, y=yL, cv=kfold, scoring=scoringA)
    meanA = np.mean(resultsAmy['test_accuracy']) #mere, izracunane s pomocjo cross_validate funkcije, za vsak k posebej
    meanA1my.append(meanA)
    print("For k = :" , k)
    print("Accuracy:")
    print(meanA)
    meanP = np.mean(resultsAmy['test_precision'])
    print("Precision:")
    meanP1my.append(meanP)
    print(meanP)
    meanR = np.mean(resultsAmy['test_recall'])
    print("Recall:")
    meanR1my.append(meanR)
    print(meanR)
    meanF = np.mean(resultsAmy['test_f1_score'])
    print("F1:")
    meanF1my.append(meanF)
    print(meanF)


print("Median of all measures acc, Pr, Rec, F1 for k = 1,..., 10:") #Povprecje mer iz prejsnjega izracuna
print(sum(meanA1my)/(len(meanA1my)))
print(sum(meanP1my)/(len(meanP1my)))
print(sum(meanR1my)/(len(meanR1my)))
print(sum(meanF1my)/(len(meanF1my)))

print("Old results:") #stari rezultat acc od k =1, ..., 10
print(k_scoresEMlc)
print(k_scoresEMstdlcresnet)
print("Old mean acc:")
print(sum(k_scoresEMlc)/(len(k_scoresEMlc)))  #povprecje od vseh stare acc za k =1, ..., 10
print("Old mean std:")
print(sum(k_scoresEMstdlcresnet)/(len(k_scoresEMstdlcresnet)))  #povprecje od vseh stare acc za k =1, ..., 10



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



mmscaler = preprocessing.MinMaxScaler()    #data leakage? no, because i us cross-validation
Xminmaxl1 = mmscaler.fit_transform(Xmy)
#print(Xminmaxb1)

"""
print("PM kNN, without LMNN:")
meanA1m = []
meanP1m = []
meanR1m = []
meanF1m = []


#brezLMNN
k_scoresP1bl1 = []
k_scoresP1stdbl1 = []
for k in range(1,11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, leaf_size=3, metric='pyfunc', metric_params={"func":poincare_distance}, n_jobs=3)	#to add new metric: metric='pyfunc'
    #for testing
    #knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='brute', metric='euclidean')    #algorithm='ball_tree' - for a custom metric
    #fit the model
    knn.fit(((Xminmaxl1)),yL)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knn, ((Xminmaxl1)), yL, cv=10, scoring='accuracy')
    k_scoresP1bl1.append(scores.mean())
    k_scoresP1stdbl1.append(scores.std())
    #for UNBALANCED DATASET
    scoringA = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score, average='macro'),
           'recall' : make_scorer(recall_score, average='macro'), 
           'f1_score' : make_scorer(f1_score, average='macro')}

    kfold = StratifiedKFold(n_splits=10) #if it is: kfold = KFold(n_splits=10, random_state=42)-->error: Setting a random_state has no effect since shuffle is False. You should leave random_state to its default (None), or set shuffle=True.
    resultsAm = cross_validate(estimator=knn, X=((Xminmaxl1)), y=yL, cv=kfold, scoring=scoringA)
    meanA = np.mean(resultsAm['test_accuracy']) #mere, izracunane s pomocjo cross_validate funkcije, za vsak k posebej
    meanA1m.append(meanA)
    print("For k = :" , k)
    print("Accuracy:")
    print(meanA)
    meanP = np.mean(resultsAm['test_precision'])
    print("Precision:")
    meanP1m.append(meanP)
    print(meanP)
    meanR = np.mean(resultsAm['test_recall'])
    print("Recall:")
    meanR1m.append(meanR)
    print(meanR)
    meanF = np.mean(resultsAm['test_f1_score'])
    print("F1:")
    meanF1m.append(meanF)
    print(meanF)


print("Median of all measures acc, Pr, Rec, F1 for k = 1,..., 10:") #Povprecje mer iz prejsnjega izracuna
print(sum(meanA1m)/(len(meanA1m)))
print(sum(meanP1m)/(len(meanP1m)))
print(sum(meanR1m)/(len(meanR1m)))
print(sum(meanF1m)/(len(meanF1m)))

print("Old results:") #stari rezultat acc od k =1, ..., 10
print(k_scoresP1bl1)
print(k_scoresP1stdbl1)

print("Old mean acc:")
print(sum(k_scoresP1bl1)/(len(k_scoresP1bl1)))  #povprecje od vseh stare acc za k =1, ..., 10
print("Old mean std:")
print(sum(k_scoresP1stdbl1)/(len(k_scoresP1stdbl1)))  #povprecje od vseh stare acc za k =1, ..., 10


"""


k_train, k_test, max_iter = 3, 3, 180	#n_components=X.shape[1], za bpp:640

# Instantiate the metric learner
lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter)

# Train the metric learner
#lmnn.fit(Xminmaxl1,yL)
lmnn.fit(Xmy,yL)

print("PM kNN, with LMNN")
meanA1myL = []
meanP1myL = []
meanR1myL = []
meanF1myL = []

k_scoresP1l1m = []
k_scoresP1stdl1m = []
for k in range(1,11):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k, leaf_size=3, metric='pyfunc', metric_params={"func":poincare_distance}, n_jobs=3)	#to add new metric: metric='pyfunc'
    
    knn.fit(mmscaler.fit_transform(lmnn.transform((Xmy))),yL)
    #cross validation-------------------------------
    scores = cross_val_score(knn, mmscaler.fit_transform(lmnn.transform((Xmy))), yL, cv=10, scoring='accuracy')
    k_scoresP1l1m.append(scores.mean())
    k_scoresP1stdl1m.append(scores.std())
    #for UNBALANCED DATASET
    scoringA = {'accuracy' : make_scorer(accuracy_score),
           'precision' : make_scorer(precision_score, average='macro'),
           'recall' : make_scorer(recall_score, average='macro'), 
           'f1_score' : make_scorer(f1_score, average='macro')}

    kfold = StratifiedKFold(n_splits=10) 
    resultsAmyL = cross_validate(estimator=knn, X= mmscaler.fit_transform(lmnn.transform((Xmy))), y=yL, cv=kfold, scoring=scoringA)
    meanA = np.mean(resultsAmyL['test_accuracy']) 
    meanA1myL.append(meanA)
    print("For k = :" , k)
    print("Accuracy:")
    print(meanA)
    meanP = np.mean(resultsAmyL['test_precision'])
    print("Precision:")
    meanP1myL.append(meanP)
    print(meanP)
    meanR = np.mean(resultsAmyL['test_recall'])
    print("Recall:")
    meanR1myL.append(meanR)
    print(meanR)
    meanF = np.mean(resultsAmyL['test_f1_score'])
    print("F1:")
    meanF1myL.append(meanF)
    print(meanF)


print("Median of all measures acc, Pr, Rec, F1 for k = 1,..., 10:") #Povprecje mer iz prejsnjega izracuna
print(sum(meanA1myL)/(len(meanA1myL)))
print(sum(meanP1myL)/(len(meanP1myL)))
print(sum(meanR1myL)/(len(meanR1myL)))
print(sum(meanF1myL)/(len(meanF1myL)))

print("Old results:") #stari rezultat acc od k =1, ..., 10
print(k_scoresP1l1m)
print(k_scoresP1stdl1m)
print("Old mean acc:")
print(sum(k_scoresP1l1m)/(len(k_scoresP1l1m)))  #povprecje od vseh stare acc za k =1, ..., 10
print("Old mean std:")
print(sum(k_scoresP1stdl1m)/(len(k_scoresP1stdl1m)))  #povprecje od vseh stare acc za k =1, ...






