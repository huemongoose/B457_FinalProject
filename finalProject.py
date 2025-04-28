import os
import xml.etree.ElementTree as ET
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV









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
        l=0
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

        hog = cv2.HOGDescriptor(_winSize=(64, 64),
                        _blockSize=(12, 12),
                        _blockStride=(4, 4),
                        _cellSize=(4, 4),
                        _nbins=9)
        if l !=3:

            transformed = cv2.flip(img, 1)
            rows, cols = transformed.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            shifted = cv2.warpAffine(rotated, M, (cols, rows))
            shifted = cv2.equalizeHist(shifted)
            feature = hog.compute(shifted)
            hogFeatures.append(feature)
            labelsLst.append(l)


            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            shifted = cv2.warpAffine(rotated, M, (cols, rows))
            shifted = cv2.equalizeHist(shifted)
            feature = hog.compute(shifted)
            hogFeatures.append(feature)
            labelsLst.append(l)

        if l == 1:
            transformed = cv2.flip(img, 1)
            rows, cols = transformed.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), -10, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            shifted = cv2.warpAffine(rotated, M, (cols, rows))
            shifted = cv2.equalizeHist(shifted)
            feature = hog.compute(shifted)
            hogFeatures.append(feature)
            labelsLst.append(l)
        if l == 2:
            transformed = cv2.flip(img, 1)
            rows, cols = transformed.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), -20, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows))
            M = np.float32([[1, 0, 5], [0, 1, 0]])
            shifted = cv2.warpAffine(rotated, M, (cols, rows))
            shifted = cv2.equalizeHist(shifted)
            feature = hog.compute(shifted)
            hogFeatures.append(feature)
            labelsLst.append(l)






        img = cv2.equalizeHist(img)
    
        feature = hog.compute(img)
        hogFeatures.append(feature)
    return hogFeatures,labelsLst

def KNN(x, y):
    scaler = StandardScaler()

    xTrain, xTest, yTrain, yTest = train_test_split(x, y,test_size=0.25,random_state=42)

    X_scaled = scaler.fit_transform(xTrain)
    xTest_scaled = scaler.transform(xTest)

    weights = {0:10,1:10,2:10,3:1}
    clf = SVC(kernel='linear',class_weight=weights,C=0.1, gamma=.001)
    clf.fit(X_scaled,yTrain)
    prediction = clf.predict(xTest_scaled)
    print(confusion_matrix(yTest,prediction))
    print(classification_report(yTest,prediction))

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
predictSVM(file1,model, scaler)#0
predictSVM(file2,model, scaler)#3

predictSVM(file3,model, scaler)#3
predictSVM(file4,model, scaler)#1
predictSVM(file5,model, scaler)#1
predictSVM(file7,model, scaler)#3
predictSVM(file8,model, scaler)#2











