import os
import random
import cv2 as cv
import pandas as pd
import numpy as np
from skimage import feature as ft
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

dim_hog = (256, 256)
low = np.array([29, 43, 46], dtype="uint8")
high = np.array([80, 255, 255], dtype="uint8")


def to_cls(cls):
    if cls == 'Black-grass':
        return 0
    if cls == 'Charlock':
        return 1
    if cls == 'Cleavers':
        return 2
    if cls == 'Common Chickweed':
        return 3
    if cls == 'Common wheat':
        return 4
    if cls == 'Fat Hen':
        return 5
    if cls == 'Loose Silky-bent':
        return 6
    if cls == 'Maize':
        return 7
    if cls == 'Scentless Mayweed':
        return 8
    if cls == 'Shepherds Purse':
        return 9
    if cls == 'Small-flowered Cranesbill':
        return 10
    if cls == 'Sugar beet':
        return 11


def for_class(num):
    if num == 0:
        return 'Black-grass'
    if num == 1:
        return 'Charlock'
    if num == 2:
        return 'Cleavers'
    if num == 3:
        return 'Common Chickweed'
    if num == 4:
        return 'Common wheat'
    if num == 5:
        return 'Fat Hen'
    if num == 6:
        return 'Loose Silky-bent'
    if num == 7:
        return 'Maize'
    if num == 8:
        return 'Scentless Mayweed'
    if num == 9:
        return 'Shepherds Purse'
    if num == 10:
        return 'Small-flowered Cranesbill'
    if num == 11:
        return 'Sugar beet'


class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X, y)
        print(grid_search.best_params_, grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


data_dir = 'train/'
classes = os.listdir(data_dir)
classes.sort()


class BOW(object):
    def __init__(self, k):
        self.feature_detector = cv.SIFT_create()
        self.descriptor_extractor = cv.SIFT_create()
        self.k = k

    def path(self, cls, i):
        path_pre = data_dir + cls + '/'
        imgname = os.listdir(path_pre)
        imgname.sort()
        return path_pre + imgname[i]

    def fit(self):
        flann_params = dict(algorithm=1, tree=5)
        flann = cv.FlannBasedMatcher(flann_params, {})
        bow_kmeans_trainer = cv.BOWKMeansTrainer(self.k)

        length = 30
        for i in range(length):
            for j in range(12):
                item = self.sift_descriptor_extractor(self.path(classes[j], i))
                if item is not None:
                    bow_kmeans_trainer.add(item)

        voc = bow_kmeans_trainer.cluster()
        print(type(voc), voc.shape)

        self.bow_img_descriptor_extractor = cv.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
        self.bow_img_descriptor_extractor.setVocabulary(voc)

        traindata, trainlabels = [], []
        for i in range(200):
            for j in range(12):
                item = self.bow_descriptor_extractor(self.path(classes[j], i))
                if item is not None:
                    traindata.extend(item)
                    trainlabels.append(j)

        print(np.array(traindata).shape)
        self.svm = SVC(C=10, kernel='rbf')
        self.svm.fit(np.array(traindata), np.array(trainlabels))

    def predict(self, img_path_pre):
        files = os.listdir(img_path_pre)
        files.sort()
        data = list()
        for f in files:
            item = self.bow_descriptor_extractor(img_path_pre + '/' + f)
            if item is not None:
                data.append(item.ravel())
            else:
                data.append(np.zeros(self.k).ravel())
        print(np.array(data).shape)
        res = self.svm.predict(data)
        return files, res

    def sift_descriptor_extractor(self, img_path):
        im = cv.imread(img_path, 1)
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        im = cv.inRange(im, low, high)
        return self.descriptor_extractor.compute(im, self.feature_detector.detect(im))[1]

    def bow_descriptor_extractor(self, img_path):
        im = cv.imread(img_path, 1)
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        im = cv.inRange(im, low, high)
        return self.bow_img_descriptor_extractor.compute(im, self.feature_detector.detect(im))


bow = BOW(120)
bow.fit()
test_id, Y_test = bow.predict('test')

test_id = np.array(test_id)
y_pred = list()
for cls in Y_test:
    y = for_class(cls)
    y_pred.append(y)
# print(len(y_pred))
y_pred = np.array(y_pred)
ResultData = pd.DataFrame(np.hstack((test_id.reshape(-1, 1), y_pred.reshape(-1, 1))), index=range(len(Y_test)),
                          columns=['file', 'species'])
ResultData.to_csv("submission_svm4th.csv", index=False)
