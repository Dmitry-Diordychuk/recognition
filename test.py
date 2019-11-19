import cv2
import numpy
import matplotlib.pyplot as plt

img1 = cv2.imread('AdobePanoramasDataset/yard-00.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

#Показать ключевые точки
#i=cv2.drawKeypoints(img1,kp1,None)
#plt.imshow(i)
#plt.show()

img2 = cv2.imread('AdobePanoramasDataset/yard-01.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(img2, None)


match = cv2.BFMatcher()
matches = match.knnMatch(des1,des2,k=2) #Найти два лучших совпадения для каждой точки
#Сохранить точку, только если первое совпадение намного лучше второго (расстояние между дескрипторами меньше)
good_matches = numpy.array([m1 for m1, m2 in matches if m1.distance < m2.distance/3])

#Показать матчи ключевых точек
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
plt.imshow(img3)
plt.show()
