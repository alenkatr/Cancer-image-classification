#rmycnnpPMskin.py
#
#januar 2024
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
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler  
from sklearn.model_selection import train_test_split

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





data_dir = 'MelanomaC/train'

list_imgsL = glob.glob(data_dir + "/**/*.jpg")

print(f"There are {len(list_imgsL)} images in the dataset {data_dir}")

#create dataloader with required transforms 

tc = transforms.Compose([
	transforms.Grayscale(),#from RGB to grayscale images
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
modelPMcancer12 = Net(100)
modelPMcancer12.eval()


print(len(dloader))
#print(dloader)


# Select the desired layer-------to get attributes from last layer in CNN model------------------------------
layer = modelPMcancer12._modules.get('dropout')
outputs1 = []


# Select the desired layer-------to get attributes from last layer in CNN model------------------------------
#copy features from penultimate layer (model, input, output)
def copy_features(m, i, o):
    o = o[:, :].detach().numpy().tolist()  #odstranila 0, 0 , ker IndexError: too many indices for tensor of dimension 2
    outputs1.append(o)

#Define and attach hook to the penulimate layer
_ = layer.register_forward_hook(copy_features)


# Generate image's features for all images in dloader and saves them in the list outputs
#---sth wrong -- with each iteration outputs are 2x bigger*??????????????????????????????=because of for loop!!!!
for X, y in dloader:
    _ = modelPMcancer12(X)
print(len(outputs1))


# flatten list of features to remove batches!!!!!!!!!!!!!!----in numpy array....
list_embC12 = [item for sublist in outputs1 for item in sublist]

print(len(list_embC12))    #vseh features je 643
print(np.array(list_embC12[0]).shape)   #dolžina ene feature je 512

Xc1122 = np.array(list_embC12)
print(f"Mn X {Xc1122.shape}")
#print(Xa1[0])





filenameL1 = os.listdir('MelanomaC/train/')
#print(filenameL)

yL122 = []
for i in range(1, len(list_imgsL)+1): #len(list_embeddings)
    sample_idx = i#torch.randint(len(test_dataset), size=(1,)).item()
    img, label = image_datasetsL[sample_idx-1]
    yL122.append(filenameL1[label])   #extend()
#print(zz)
#print(f"[{', '.join(yy1)}]")
print(len(yL122))



def Poincare_P_dist(row1, row2, q):
    #print('len(row1) = ',len(row1))
    #print('len(row1) = ',len(row1))
    distance = 0.0
    d = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    #for p in np.linspace(1,20,11):
    d = np.sum(np.abs(row1-row2)**q)**(1/q)  #||p-q||=d.sum((row1 - row2)*(row1 - row2))--->distance.minkowski(row1, row2, p=3)
    #d = d.sum()

    norm1 = np.sum(np.abs(row1)**q)**(1/q)
    #norm1 = norm1.sum()        #||p||=sum.(row1 * row1)----> np.sum(np.abs(row1)**p)**(1/p)

    norm2 = np.sum(np.abs(row2)**q)**(1/q)
    #norm2 = norm2.sum()        #||q||

    m = 1 + 2*d*d/((1-norm1*norm1)*(1-norm2*norm2))
    distance = np.log(m + np.sqrt(m*m - 1))

    return(distance)

"""
X_train, X_test, y_train, y_test = train_test_split(Xc1122, yL122, test_size=0.8, random_state=0)


#find best parameters k and q
#scores = cross_val_score(knn, X1lc, ylc, cv=10, scoring='accuracy')
#Parametri: p iz Lp norme v poincare metriki, namesto evklidske norme
#k iz kNN


tuned_parametersC = [{'n_neighbors': [1,2, 3, 5, 7, 10], "p":[1, 2, 3, 5, 7, 10]}]  #ne daj not k=1, ker nima smmisla

scores = ['accuracy']
kNN = neighbors.KNeighborsClassifier(metric='pyfunc', metric_params={"func":Poincare_P_dist,"p":p}, n_jobs=2)
print("# Tuning hyper-parameters for accuracy")
clfC = GridSearchCV(kNN, tuned_parametersC)
clfC.fit(X_train, y_train)


print("Best parameters set found on development set:")
print()
print(clfC.best_params_)
print()
print("Detailed classification report:")
y_true, y_pred = y_test, clfC.predict(X_test)
print(classification_report(y_true, y_pred))



##################################################
#
#
#fixing k=3 and different metric in PM
#
#
i_scores = []
#for i in range(1,11):
for p in np.linspace(1,10,10):
    knnC = neighbors.KNeighborsClassifier(n_neighbors=3, leaf_size=3,algorithm='ball_tree', metric='pyfunc', metric_params={"func":Poincare_P_dist,"p":p}, n_jobs=4)
    #ball_tree makes better classification accuracy!!!!!
    #fit the model
    knnC.fit(Xc1122,yL122)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knnC, Xc1122, yL122, cv=10, scoring='accuracy')
    i_scores.append(scores.mean())
    print("For p =", p)
print("For p=", p, "and k=",3,"the accuracy is:", i_scores)

"""


"""
#different norm q in Poincare metric
js_scores = []
#for j in range(1,11):
for q in np.linspace(1,10,10):
    knnCs = neighbors.KNeighborsClassifier(n_neighbors=2, leaf_size=3,algorithm='ball_tree', metric='pyfunc', metric_params={"func":Poincare_P_dist,"q":q}, n_jobs=4) #tu sem spreminjala rocno K=2,3,5,7
    #ball_tree makes better classification accuracy!!!!!
    #fit the model
    knnCs.fit(Xc1122,yL122)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knnCs, Xc1122,yL122, cv=10, scoring='accuracy')
    js_scores.append(scores.mean())
    print("For q =", q)
print("For q=", 1,...,10, "and k=",2,"the accuracy is:", js_scores)
"""


#for compute Chebyshev distance
#for p metric for different p....code from sklearn
#from sklearn.metrics import pairwise_distances
#from scipy.spatial import distance
from scipy.spatial.distance import chebyshev
from scipy.spatial.distance import minkowski
#che = distance.cdist(XA, XB, 'chebyshev')

t_scores1ls = []
#for j in range(1,11):
for k in range(1,11):
    #distance = pairwise_distances(Xc112,Xc112, metric='minkowski', p=q)
    knnC1ls = neighbors.KNeighborsClassifier(n_neighbors=k, leaf_size=3,algorithm='ball_tree',metric='chebyshev', n_jobs=3) #t
    #ball_tree makes better classification accuracy!!!!!
    #fit the model
    knnC1ls.fit(Xc1122,yL122)
    #cross validation-------------------------------after knn
    scores = cross_val_score(knnC1ls, Xc1122,yL122, cv=10, scoring='accuracy')
    t_scores1ls.append(scores.mean())
    print("For k =", k)
print("For k =", 1,...,10, "chebyshev metric","the accuracy is:", t_scores1ls)



























