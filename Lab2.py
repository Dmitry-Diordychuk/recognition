import cv2
import numpy
import matplotlib.pyplot as plt

CPU = 4

import os
files = os.listdir("C:\Python\.vscode\AdobePanoramasDataset")
path = "C:\Python\.vscode\AdobePanoramasDataset/"
resultList = list()
class MyImage(object):
    """docstring"""
 
    def __init__(self, img, name):
        """Constructor"""
        self.img = img
        self.name = name

imgList = []
for file_n in files:
    newImg = MyImage(cv2.imread(path+file_n), file_n)
    imgList.append(newImg)


from itertools import combinations
def pairwiseCombination(iterable):
    a = iter(iterable)
    return combinations(a, 2)

surf = cv2.xfeatures2d.SURF_create(upright=1)#SIFT_create()
match = cv2.BFMatcher()
def matchImages(img1,img2): 
    kp1, des1 = surf.detectAndCompute(img1.img, None)
    kp2, des2 = surf.detectAndCompute(img2.img, None)

    matches = match.knnMatch(des1,des2, k=2) #Найти два лучших совпадения для каждой точки
    #Сохранить точку, только если первое совпадение намного лучше второго (расстояние между дескрипторами меньше)
    good_matches = numpy.array([m1 for m1, m2 in matches if m1.distance < m2.distance/3])

    if good_matches.size > 4:
        print(img1.name + " matched with " + img2.name)
        #resultList.append(img1.name + " matched with " + img2.name)

import copy
from multiprocessing import Pool
import time
if __name__ == "__main__":
    #Parallel(n_jobs=CPU, prefer="threads" , verbose=50)(delayed(matchImages)(img1, img2) for img1,img2 in pairwiseCombination(imgList))
    p = Pool(CPU)
    #now = time.time()
    p.starmap(matchImages, pairwiseCombination(imgList) )
    #print(time.time()-now)

    resultList.sort()
    with open('result.txt', 'w') as f:
        for item in resultList:
            f.write("%s\n" % item)


