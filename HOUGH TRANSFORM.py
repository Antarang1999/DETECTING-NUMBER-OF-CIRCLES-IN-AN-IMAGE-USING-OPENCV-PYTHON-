import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread('')   # image in which circles are to be detected
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)

cv2.imwrite("C:\Users\win\Desktop\gray.jpg", gray) #where you want to save your file    
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.imread('C:\Users\win\Desktop\gray.jpg')

kernel = np.ones((5,5),np.float32)/25
smooth = cv2.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(smooth),plt.title('Averaging')
plt.xticks([]), plt.yticks([])

cv2.imwrite("C:\Users\win\Desktop\smooth.jpg",smooth)
plt.show()



img = cv.imread("C:\Users\win\Desktop\smooth.jpg",0)
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
cv2.imwrite("C:\Users\win\Desktop\thresh1.jpg",thresh1)


img = cv2.imread('C:\Users\win\Desktop\thresh1.jpg',0)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

cv2.imwrite("C:\Users\win\Desktop\sobel.jpg",sobelx)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


img = cv2.imread('C:\Users\win\Desktop\sobel.jpg',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1.6,10,param1=50,param2=30,minRadius=0,maxRadius=10)
#if the image doesn't recognize all the circles these parameters can be can be change to achieve the required output.


circles = np.uint16(np.around(circles))
#loop to detect all the circles in the image
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.imwrite("C:\Users\win\Desktop\cimg.jpg",cimg)
# this is ur outpt file which contains the detected circle,if a counter is used in the for loop we get the number of circles detected in the image.   

cv2.waitKey(0)
cv2.destroyAllWindows()