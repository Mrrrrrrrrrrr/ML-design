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

# 图片预处理后将其固定为dim_hog大小并提取特征
dim_hog = (256, 256)


# 将类名转换为数字，方便SVM分类
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


# 将预测结果转为类名并输出至csv文件中
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


# 网格搜索函数，利用该函数进行任意模型的网格搜索，并确定超参
class grid():
    def __init__(self, model):
        self.model = model

    def grid_get(self, X, y, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X, y)
        print(grid_search.best_params_, grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score', 'std_test_score']])


# 颜色提取arrary，low和high分别界定了需要提取颜色的上下限
# 该颜色表示形式为HSV颜色模型
low = np.array([29, 43, 46], dtype="uint8")
high = np.array([80, 255, 255], dtype="uint8")

# 读取训练集数据
data_dir = 'train/'
classes = os.listdir(data_dir)
classes.sort()
data_train = list()
clslist = list()
# 按照类名依次读取
for cls in classes:
    files = os.listdir(data_dir + cls)
    for f in files:
        # 先读取原彩色图像，读取后为一个BGR三通道图像
        img = cv.imread(data_dir + cls + "/" + f, cv.IMREAD_COLOR)
        # 想要舍弃分辨率过低的图像，因为低分辨率图像有可能会使训练误差变大，界定阈值为（128，128）
        if len(img) >= 128 and len(img[0]) >= 128:
            # 将原图颜色转为HSV形式
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            # 分割出绿色部分，分割后为黑白图像
            img = cv.inRange(img, low, high)
            # cv.namedWindow('img', 0)
            # cv.resizeWindow('img', 800, 600)
            # cv.imshow('img', img)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # 将处理好的图片放入clslist列表，cls为图片的类名
            clslist.append([cls, img])
    # 类中所有图像读取完后，若图像总数过少，可能需要进行数据增强
    # 此处对类中图像少于200张的类进行增强
    if len(clslist) <= 200:
        # 首先打乱类中图像
        random.shuffle(clslist)
        # 对前60张图像进行逆时针旋转30°的处理
        for i in range(60):
            rows, cols = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append([cls, rotated])
        # 对接下来30张图像进行水平翻转
        for i in range(60, 90):
            fl0 = cv.flip(clslist[i][1], 0)
            clslist.append([cls, fl0])
        # 对其后50张图像进行垂直翻转
        for i in range(70, 120):
            fl1 = cv.flip(clslist[i][1], 1)
            clslist.append([cls, fl1])
        # 对其后70张图像进行逆时针135°旋转处理
        for i in range(80, 150):
            rows, cols = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 135, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append([cls, rotated])
    # 对处于200到300个数据的类做类似处理
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
    # 将该类中所有图像加入训练数据中
    data_train = data_train + clslist
    clslist.clear()
    print(cls + " Complete")

# 对训练数据进行洗牌
random.shuffle(data_train)
# [0, 4867)
del clslist, classes, data_dir

# 读取测试数据，方便与训练数据一起预处理
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

# data_all代表训练集与测试集之和
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

# for i in range(2):
#     print(i)
#     print(type(feature_sift[0][0][i]))
#     print(feature_sift[0][0][i].pt)
#     print(feature_sift[0][0][i].size)
#     print(feature_sift[0][0][i].angle)
#     print(feature_sift[0][0][i].octave)

# 对所有图像进行预处理
for img in data_all:
    # 高斯模糊，高斯核大小3*3
    img[1] = cv.GaussianBlur(img[1], (3, 3), 0)
    # 拉普拉斯算子用于边缘检测
    img[1] = cv.Laplacian(img[1], cv.CV_64F)
    img[1] = cv.convertScaleAbs(img[1])
    # img[1] = cv.Canny(img[1], 32, 64)
    # 将处理好的图像固定为dim_hog大小
    img[1] = cv.resize(img[1], dim_hog)

# for i in range(0, 300):
#     cv.namedWindow('img', 0)
#     cv.resizeWindow('img', 800, 600)
#     cv.imshow('img', data_all[i][1])
#     cv.waitKey(0)
#     cv.destroyAllWindows()

# 预设两个列表分别用于储存训练集和测试集图像利用HOG特征提取后的数据
features_hog_train = list()
features_hog_test = list()
# 提取训练集HOG特征
for img in data_train:
    # 采用16个bin，每个cell的大小为[32,32]，每个block有2*2个cell，visualize为False，不需要可视化
    hog_feature = ft.hog(img[1], orientations=16, pixels_per_cell=[32, 32], cells_per_block=[2, 2], visualize=False)
    features_hog_train.append(hog_feature)
# 提取测试集HOG特征
for img in data_test:
    hog_feature = ft.hog(img[1], orientations=16, pixels_per_cell=[32, 32], cells_per_block=[2, 2], visualize=False)
    features_hog_test.append(hog_feature)
# plt.imshow(features_hog_test[0][1])
# plt.show()

# 将提取到的HOG特征作为训练用特征
X_train = np.array(features_hog_train)
X_test = np.array(features_hog_test)
# Y_train表示训练集的标签
Y_train = list()
for item in data_train:
    Y_train.append(to_cls(item[0]))
Y_train = np.array(Y_train)
Y_train = Y_train.ravel()
print(X_train.shape)
print(Y_train.shape)
# 由于HOG特征维数太多，影响训练，于是用PCA主成分分析进行降维
pca = PCA(n_components=0.8)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print(X_train.shape)

# 以下为利用网格搜索进行超参调试的过程

# param_grid = {'C': [1, 10, 100], 'kernel': ["rbf", "poly"]}
# grid(SVC()).grid_get(X_train, Y_train, param_grid)
# # C=10, kernel='rbf'

# param_grid = {'n_estimators': [100, 300, 500]}
# grid(RandomForestClassifier()).grid_get(X_train, Y_train, param_grid)
# # n = 300 >> +na

# param_grid = {'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200], 'subsample': [0.8]}
# grid(XGBClassifier()).grid_get(X_train, Y_train, param_grid)
# # learning_rate = 0.05, n_estimators = 200

# 以下分别为利用SVM、随机森林分类、XSBoost分类器进行训练并预测的过程

svcmodel = SVC(C=10, kernel='rbf')
svcmodel.fit(X_train, Y_train)
Y_test = svcmodel.predict(X_test).reshape(-1, 1)

# rf = RandomForestClassifier(n_estimators=300)
# rf.fit(X_train, Y_train)
# Y_test = rf.predict(X_test).reshape(-1, 1)

# xgb = XGBClassifier(learning_rate=0.05, n_estimators=200)
# xgb.fit(X_train, Y_train)
# Y_test = xgb.predict(X_test).reshape(-1, 1)
#

# 接下来打印预测结果并形成submission.csv文件
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
ResultData.to_csv("submission_svm1st_nopro.csv", index=False)
