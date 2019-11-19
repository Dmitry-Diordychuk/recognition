import cv2
import numpy
import matplotlib.pyplot as plt
import skimage.data
from skimage import io


k=1
while k < 76:
    path = "C:/dataset_segmentation/"+ str(k) +".jpg" #str(k)

    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    imgarray = numpy.array(img)

    imgarray[imgarray<200] = 1.0
    imgarray[imgarray>=56] = 0.0

    i=0
    while i < 3:
        array = numpy.array([[1]*21*i]*21*i)
        imgarray = cv2.morphologyEx(imgarray, cv2.MORPH_OPEN, array)
        array = numpy.array([[1]*63*i]*63*i)
        imgarray = cv2.morphologyEx(imgarray, cv2.MORPH_CLOSE, array)
        i = i + 1

    plt.imshow(imgarray, cmap='Greys') #Needs to be in row,col order
    plt.savefig(str(k) + ".jpg", cmap='Greys')
    k += 1