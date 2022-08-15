# # !/usr/bin/env python
# # -*- coding:utf-8 -*-
#
# from matplotlib import pylab as plt
#
# import numpy as np
# import cv2 as cv
#
# # 读入图像
# img_name_1 = "./data/0/0.png"
# img_name_2 = "./data/0/180.png"
#
# img_1 = cv.imread(img_name_1)
# img_2 = cv.imread(img_name_2)
#
# # 转换为灰度图像：图像处理一般都是灰度图像
# gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
# gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)
#
# # 实例化SIFT算子
# sift = cv.SIFT_create()
#
# # 分别对两张图像进行SIFT检测
# kp_1, des_1 = sift.detectAndCompute(img_1, None)
# kp_2, des_2 = sift.detectAndCompute(img_2, None)
#
# # 显示特征点
# img_res_1 = cv.drawKeypoints(img_1, kp_1, gray_1, color=(255, 0, 255))
# img_res_2 = cv.drawKeypoints(img_2, kp_2, gray_2, color=(0, 0, 255))
# cv.imshow("SIFT_image_1", img_res_1)
# cv.imshow("SIFT_image_2", img_res_2)
#
# # BFMatcher算法匹配
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des_1, des_2, k=2)
#
# # 筛选优质的匹配点
# ratio = 0.75
# good_features = []
# for m, n in matches:
#     if m.distance < ratio * n.distance:
#         good_features.append([m])
#
# # 将匹配的特征点绘制在一张图内
# img_res = cv.drawMatchesKnn(img_1, kp_1, img_2, kp_2, good_features, None, flags=2)
# cv.imshow("BFmatch", img_res)
#
# cv.waitKey(0)
# cv.destroyAllWindows()
# !/usr/bin/env python
# -*- coding:utf-8 -*-

from matplotlib import pylab as plt

import numpy as np
import cv2 as cv

# 读入图像
img_name_1 = "./data/0/0.png"
img_name_2 = "./data/0/180.png"

img_1 = cv.imread(img_name_1)
img_2 = cv.imread(img_name_2)

# 转换为灰度图像：图像处理一般都是灰度图像
gray_1 = cv.cvtColor(img_1, cv.COLOR_BGR2GRAY)
gray_2 = cv.cvtColor(img_2, cv.COLOR_BGR2GRAY)

# 实例化ORB算子
orb = cv.ORB_create()
kp_1, des_1 = orb.detectAndCompute(img_1, None)
kp_2, des_2 = orb.detectAndCompute(img_2, None)

# BFMatcher算子匹配
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_1, des_2)
matches = sorted(matches, key=lambda x: x.distance)

img_3 = cv.drawMatches(img_1, kp_1, img_2, kp_2, matches[:180], img_2, flags=2)
cv.imshow("result", img_3)

cv.waitKey()
cv.destroyAllWindows()

