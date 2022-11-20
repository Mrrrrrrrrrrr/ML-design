import os
import random
import cv2 as cv

data_dir = 'train/'
classes = os.listdir(data_dir)
data = list()
clslist = list()
for cls in classes:
    files = os.listdir(data_dir + cls)
    for f in files:
        img = cv.imread(data_dir + cls + "/" + f)
        clslist.append((cls, img))
    if len(clslist) <= 300:
        random.shuffle(clslist)
        for i in range(60):
            rows, cols, channel = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append(rotated)
        for i in range(60, 140):
            fl0 = cv.flip(clslist[i][1], 0)
            clslist.append(fl0)
        for i in range(140, 220):
            fl1 = cv.flip(clslist[i][1], 1)
            clslist.append(fl1)
        for i in range(220, 300):
            fl2 = cv.flip(clslist[i][1], -1)
            clslist.append(fl2)
    if len(clslist) > 300 and len(clslist) <= 400:
        random.shuffle(clslist)
        for i in range(30):
            rows, cols, channel = clslist[i][1].shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
            rotated = cv.warpAffine(clslist[i][1], M, (cols, rows))
            clslist.append(rotated)
        for i in range(30, 70):
            fl0 = cv.flip(clslist[i][1], 0)
            clslist.append(fl0)
        for i in range(70, 110):
            fl1 = cv.flip(clslist[i][1], 1)
            clslist.append(fl1)
        for i in range(110, 150):
            fl2 = cv.flip(clslist[i][1], -1)
            clslist.append(fl2)
    # print(len(clslist))
    data = data + clslist
    clslist.clear()

random.shuffle(data)
print(len(data))