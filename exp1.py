"""
Task [I] - Demonstrating how to compute the histogram of an image using 4 methods.
(1). numpy based
(2). matplotlib based
(3). opencv based
(4). do it myself (DIY)
check the precision, the time-consuming of these four methods and print the result.
"""


import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2


# please coding here for solving Task [I].

# file_name = './canoe.tif'
# img = cv2.imread(file_name)
#
# red_hist = cv2.calcHist(img,[2],None,[256],[0.0,256.0]) #cv2的方法
#
# plt.plot(red_hist,color='r')
# plt.show()


# def histtogram(img,bins):
#     N_x = np.zeros_like(bins,dtype=np.float)
#     for (i,levels) in enumerate(bins):
#         N_x[i] = np.sum(img == levels)
#     return N_x
# file_name = './canoe.tif'
# img = cv2.imread(file_name)
# intensity_levels = np.arange(0,256,1)             #老师的代码
# red_hist = histtogram(img[2,:,:],intensity_levels)
#
# plt.plot(intensity_levels,red_hist,color='red')
# plt.show()
#
# file_name = './canoe.tif'
# img = cv2.imread(file_name)
# plt.hist(img[:,:,2].ravel(),bins=256,range=[0,256],color='r')
# plt.show()

# file_name = './canoe.tif'
# img = cv2.imread(file_name)
# plt.figure()
# plt.hist(img[2,:,:].flatten(), 256, (0,256))  #matplotlib的函数
# plt.show()

# file_name = './canoe.tif'
# img = cv2.imread(file_name)
# hist,bins = np.histogram(img[2,:,:].ravel(),256,[0,256])  #np.histogram
# plt.plot(hist,color='r')
# plt.show()











###





"""
Task [II]Refer to the link below to do the gaussian filtering on the input image.
Observe the effect of different @sigma on filtering the same image.
Try to figure out the gaussian kernel which the ndimage has used [Solution to this trial wins bonus].
https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
"""

###
#please coding here for solving Task[II]


# from scipy.ndimage import gaussian_filter
# file_name = './canoe.tif'
# img = cv2.imread(file_name)
# img_1 = gaussian_filter(img,sigma=1)
# red_hist = cv2.calcHist(img_1,[2],None,[256],[0.0,256.0])  #网上找的calchist函数
# plt.plot(red_hist,color='r')
# plt.show()
# cv2.imshow('before',img)
# cv2.imshow('after',img_1)
# cv2.waitKey()








"""
Task [III] Check the following link to accomplish the generating of random images.
Measure the histogram of the generated image and compare it to the according gaussian curve
in the same figure.
"""

###
#please coding here for solving Task[III]

# mean = (1,2)  #2维?    ???
# cov = [[1,0],[0,1]]
# x = np.random.multivariate_normal(mean,cov,(2,2),'raise') #2*2的矩阵?   ....参数没弄明白
# print(x)
# plt.hist(x.flatten(),256,(0,256))
# plt.show()

