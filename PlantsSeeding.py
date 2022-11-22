import os
import random
import cv2 as cv
import pandas as pd
import numpy as np
from skimage import feature as ft
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.decomposition import PCA

dim = (256, 256)


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


low = np.array([29, 43, 46], dtype="uint8")
high = np.array([80, 255, 255], dtype="uint8")

data_dir = 'train/'
classes = os.listdir(data_dir)
classes.sort()
data_train = list()
clslist = list()
for cls in classes:
    files = os.listdir(data_dir + cls)
    for f in files:
        img = cv.imread(data_dir + cls + "/" + f, cv.IMREAD_COLOR)
        if len(img) >= 128 and len(img[0]) >= 128:
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            img = cv.inRange(img, low, high)
            # cv.namedWindow('img', 0)
            # cv.resizeWindow('img', 800, 600)
            # cv.imshow('img', img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            clslist.append([cls, img])
    if len(clslist) <= 200:
        random.shuffle(clslist)
        for i in range(60):
            rows, cols = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append([cls, rotated])
        for i in range(60, 90):
            fl0 = cv.flip(clslist[i][1], 0)
            clslist.append([cls, fl0])
        for i in range(70, 120):
            fl1 = cv.flip(clslist[i][1], 1)
            clslist.append([cls, fl1])
        for i in range(80, 150):
            rows, cols = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 135, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append([cls, rotated])
    if len(clslist) > 200 and len(clslist) <= 300:
        random.shuffle(clslist)
        for i in range(30):
            rows, cols = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append([cls, rotated])
        for i in range(30, 70):
            fl0 = cv.flip(clslist[i][1], 0)
            clslist.append([cls, fl0])
        for i in range(70, 110):
            fl1 = cv.flip(clslist[i][1], 1)
            clslist.append([cls, fl1])
        for i in range(110, 150):
            rows, cols = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 135, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append([cls, rotated])
    print(len(clslist))
    data_train = data_train + clslist
    clslist.clear()
    print(cls + " Complete")

random.shuffle(data_train)
# [0, 4867)
del clslist, classes, data_dir

data_dir = 'test'
files = os.listdir(data_dir)
files.sort()
data_test = list()
for f in files:
    img = cv.imread(data_dir + "/" + f, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = cv.inRange(img, low, high)
    data_test.append([f, img])
print("Test Complete")
del data_dir, files
# [4867, 5661)

data_all = data_train + data_test
print(len(data_all))
print(len(data_train))
print(len(data_test))

# data_dir = 'train/Charlock'
# demo = list()
# files = os.listdir(data_dir)
# for i in range(10):
#     img = cv.imread(data_dir + "/" + files[i])
#     img = cv.resize(img, dim)
#     demo.append(('Charlock', img))

# sift = cv.SIFT_create()
# features_sift = list()
# for img in demo:
#     (kp, des) = sift.detectAndCompute(img[1], None)
#     features_sift.append((kp, des))

# for i in range(2):
#     print(i)
#     print(type(feature_sift[0][0][i]))
#     print(feature_sift[0][0][i].pt)
#     print(feature_sift[0][0][i].size)
#     print(feature_sift[0][0][i].angle)
#     print(feature_sift[0][0][i].octave)
for img in data_all:
    img[1] = cv.GaussianBlur(img[1], (3, 3), 0)
    img[1] = cv.Laplacian(img[1], cv.CV_64F)
    img[1] = cv.convertScaleAbs(img[1])
    img[1] = cv.resize(img[1], dim)

# cv.namedWindow('img', 0)
# cv.resizeWindow('img', 800, 600)
# cv.imshow('img', data_all[500][1])
# cv.waitKey(0)
# cv.destroyAllWindows()

features_hog_train = list()
features_hog_test = list()
for img in data_train:
    hog_feature = ft.hog(img[1], orientations=16, pixels_per_cell=[32, 32], cells_per_block=[2, 2], visualize=False)
    features_hog_train.append(hog_feature)
for img in data_test:
    hog_feature = ft.hog(img[1], orientations=16, pixels_per_cell=[32, 32], cells_per_block=[2, 2], visualize=False)
    features_hog_test.append(hog_feature)
# plt.imshow(features_hog[0][1])
# plt.show()
X_train = np.array(features_hog_train)
X_test = np.array(features_hog_test)
Y_train = list()
for item in data_train:
    Y_train.append(to_cls(item[0]))
Y_train = np.array(Y_train)
Y_train = Y_train.ravel()
print(X_train.shape)
print(Y_train.shape)
pca = PCA(n_components=0.8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(X_train.shape)
# param_grid = {'C': [1, 10, 100], 'kernel': ["rbf", "poly"]}
# grid(SVC()).grid_get(X_train, Y_train, param_grid)
# C=10, kernel='rbf'
svcmodel = SVC(C=10, kernel='rbf')
svcmodel.fit(X_train, Y_train)
Y_test = svcmodel.predict(X_test).reshape(-1, 1)
test_id = list()
for img in data_test:
    test_id.append(img[0])
test_id = np.array(test_id)
y_pred = list()
for cls in Y_test:
    y = for_class(cls)
    y_pred.append(y)
# print(len(y_pred))
y_pred = np.array(y_pred)
ResultData = pd.DataFrame(np.hstack((test_id.reshape(-1, 1), y_pred.reshape(-1, 1))), index=range(len(Y_test)),
                          columns=['file', 'species'])
ResultData.to_csv("submission_svm1st.csv", index=False)
