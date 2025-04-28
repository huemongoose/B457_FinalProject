import os
import xml.etree.ElementTree as ET
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier







images = 'images'
labels = 'annotations'

def classifying(images, labels):

    labelsLst = []
    hogFeatures= [] 
    for file in os.listdir(labels):
        path = os.path.join(labels,file)

        tree = ET.parse(path)
        root = tree.getroot()

        filename = root.find('filename').text
        image_path = os.path.join(images, filename)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))

        objects = root.findall('object')
        for obj in objects:
            name = obj.find('name').text.lower()
            if name == 'stop':
                l = 0
            elif name == 'trafficlight':
                l =1
            elif name == 'crosswalk':
                l =2
            elif name == 'speedlimit':
                l = 3
            labelsLst.append(l)
            break


        img = cv2.equalizeHist(img)
        hog = cv2.HOGDescriptor(_winSize=(64, 64),
                        _blockSize=(12, 12),
                        _blockStride=(4, 4),
                        _cellSize=(4, 4),
                        _nbins=9)
        feature = hog.compute(img)
        hogFeatures.append(feature)
    return hogFeatures,labelsLst

def KNN(x, y):
    scaler = StandardScaler()

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, random_state=42)

    X_scaled = scaler.fit_transform(xTrain)

    clf = SVC(kernel='linear', gamma=2)
    clf.fit(X_scaled,yTrain)
    return clf, scaler



def predictSVM(img, model,scaler):
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    img = cv2.equalizeHist(img)
    hog = cv2.HOGDescriptor(_winSize=(64, 64),
                        _blockSize=(12, 12),
                        _blockStride=(4, 4),
                        _cellSize=(4, 4),
                        _nbins=9)
    feature = hog.compute(img)
    feature = feature.reshape(1, -1)  
    feature_scaled = scaler.transform(feature)
    prediction = model.predict(feature_scaled)
    print(f"Prediction on image: {prediction}")

 
    












x,y = classifying(images,labels)

file1 = "STOP_sign.jpg"
file2 = "images/road758.png"
file3 = "images/road810.png"
file4 ="images/road8.png"
file5 = "images/road1.png"
file7 ="images/road847.png"
file8 = "images/road315.png"
model,scaler =KNN(x,y)
predictSVM(file1,model, scaler)
predictSVM(file2,model, scaler)

predictSVM(file3,model, scaler)
predictSVM(file4,model, scaler)
predictSVM(file5,model, scaler)
predictSVM(file7,model, scaler)
predictSVM(file8,model, scaler)










