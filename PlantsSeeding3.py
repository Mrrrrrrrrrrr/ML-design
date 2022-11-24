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


# 以上部分与PlantsSeeding中相同，不再作解释

# 读取训练集中的所有类名
data_dir = 'train/'
classes = os.listdir(data_dir)
classes.sort()


# 构建BOW类，用于训练与预测数据
class BOW(object):
    # 构造函数设置了两个SIFT函数，feature_detector用于寻找图像中的关键点，descriptor_extractor用于计算关键点描述符，k则是代表了K聚类分析中超参K的值
    def __init__(self, k):
        self.feature_detector = cv.SIFT_create()
        self.descriptor_extractor = cv.SIFT_create()
        self.k = k

    # 用于返回训练集cls类中第i个图像的完整地址
    def path(self, cls, i):
        path_pre = data_dir + cls + '/'
        imgname = os.listdir(path_pre)
        imgname.sort()
        return path_pre + imgname[i]

    # 训练函数，该函数内包含了BOWK聚类分析构建过程与SVM分类器的训练过程
    def fit(self):
        # 基于FLANN的匹配器，算法为KDTREE，树的个数为5
        flann_params = dict(algorithm=1, tree=5)
        flann = cv.FlannBasedMatcher(flann_params, {})
        # BagOfWords方法，将用作训练的图像分置到k个簇中
        bow_kmeans_trainer = cv.BOWKMeansTrainer(self.k)

        # length代表每个类用于BOWK训练器的图象个数
        length = 30
        for i in range(length):
            for j in range(12):
                # 提取该图像的SIFT描述符
                item = self.sift_descriptor_extractor(self.path(classes[j], i))
                # 由于极个别图像经过预处理后为一片黑色，提取不到描述符，因此将提取到的描述符加入BOWK聚类器训练
                if item is not None:
                    bow_kmeans_trainer.add(item)

        # 利用cluster函数进行视觉词汇创建
        voc = bow_kmeans_trainer.cluster()
        print(type(voc), voc.shape)

        # 初始化BOW提取器
        self.bow_img_descriptor_extractor = cv.BOWImgDescriptorExtractor(self.descriptor_extractor, flann)
        # 将视觉词汇集合构成视觉词典
        self.bow_img_descriptor_extractor.setVocabulary(voc)

        # 提取每类200张图片作为训练集
        traindata, trainlabels = [], []
        for i in range(200):
            for j in range(12):
                # 获取训练图像描述符
                item = self.bow_descriptor_extractor(self.path(classes[j], i))
                # 去除不符合的两张，共2398张
                if item is not None:
                    # 将该图像描述符加入训练特征集
                    traindata.extend(item)
                    trainlabels.append(j)

        print(np.array(traindata).shape)
        # 利用SVM分类器训练数据集
        self.svm = SVC(C=10, kernel='rbf')
        self.svm.fit(np.array(traindata), np.array(trainlabels))

    def predict(self, img_path_pre):
        # 读取测试集数据
        files = os.listdir(img_path_pre)
        files.sort()
        data = list()
        for f in files:
            item = self.bow_descriptor_extractor(img_path_pre + '/' + f)
            if item is not None:
                # 当测试图像有特征点时，将BOW提取器提取到的结果作为该图像特征加入测试数据中
                data.append(item.ravel())
            else:
                # 当无特征点，代表该图像的BOW特征都为0，即SIFT特征点所有聚类中心出现的次数都为0，则将一个k大小的全0向量加入测试数据
                data.append(np.zeros(self.k).ravel())
        print(np.array(data).shape)
        # 进行SVM分类器预测
        res = self.svm.predict(data)
        return files, res

    def sift_descriptor_extractor(self, img_path):
        # SIFT特征提取前，进行图像预处理（颜色分割）
        im = cv.imread(img_path, 1)
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        im = cv.inRange(im, low, high)
        # 返回所有SIFT特征点数据
        return self.descriptor_extractor.compute(im, self.feature_detector.detect(im))[1]

    def bow_descriptor_extractor(self, img_path):
        im = cv.imread(img_path, 1)
        im = cv.cvtColor(im, cv.COLOR_BGR2HSV)
        im = cv.inRange(im, low, high)
        # 进行BOW分析计算出每个图像的BOW特征
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
