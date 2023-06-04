#!/usr/bin/env python
# coding: utf-8

# # Q.1.a

# In[87]:


import cv2


# In[88]:


#Showing and loading images we have, we use cv2 library and following codes
img=cv2.imread('bird-migration.jpg')
cv2.imshow('bird-migration.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[89]:


#Splitting B,G,R channels of image and showing that splitted images as blue, green and red
img_B = img[:,:,0]
img_G = img[:,:,1]
img_R = img[:,:,2]

cv2.imshow('image_B',img_B)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image_G',img_G)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image_R',img_R)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[90]:


#Saving splitted B,G,R images as wanted format
status1 = cv2.imwrite('bird_blue.tif',img_B)
status2 = cv2.imwrite('bird_green.tif',img_G)
status3 = cv2.imwrite('bird_red.tif',img_R)


# # Q.1.b

# In[91]:


import matplotlib.pyplot as plt


# In[92]:


#Creating histogram tables each of grayscales
hist1 = cv2.calcHist([img_B],[0],None,[256],[0,256])
plt.subplot(121),plt.imshow(img_B,'gray')
plt.subplot(122),plt.plot(hist1)
plt.xlim([0,256])
plt.title('blue')
plt.show()

hist2 = cv2.calcHist([img_G],[0],None,[256],[0,256])
plt.subplot(221),plt.imshow(img_G,'gray')
plt.subplot(222),plt.plot(hist2)
plt.xlim([0,256])
plt.title('green')
plt.show()

hist3 = cv2.calcHist([img_R],[0],None,[256],[0,256])
plt.subplot(231),plt.imshow(img_R,'gray')
plt.subplot(232),plt.plot(hist3)
plt.xlim([0,256])
plt.title('red')
plt.show()


# # Q.1.c

# In[93]:


#Channels of image bird migration
img_B = img[:,:,0]
img_G = img[:,:,1]
img_R = img[:,:,2]

#Negatives of each grayscale images

negative_of_img_B = 1 - img_B
negative_of_img_G = 1 - img_G
negative_of_img_R = 1 - img_R


# In[94]:


#After above so easy mathematical transformation,if we want to see their negative forms, we can use cv2 again.
import cv2


# In[95]:


cv2.imshow('neg_1',negative_of_img_B)
cv2.waitKey(0)
cv2.destroyAllWindows()
#We can apply this process each of them. It can work successfully!


# # Q.1.d

# In[96]:


import cv2 #for showing images
import numpy as np #I use this for creating arrays


# In[97]:


#Creating zero matrix by using rows and columns of the image
r = img_B.shape[0]
c = img_B.shape[1]

img_thres = np.zeros((r,c))
n_pix = 0

#I created a thresold that counts black pixel because the birds are black in the image.
for y in range(0,r):
    for x in range(0,c):
        pixel = img_B[y,x]
        if pixel > 60:
            n_pix = pixel
        else:
            n_pix = 0
        img_thres[y,x] = n_pix

#Here you can see easily birds and count.        
cv2.imshow('img',img_thres)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Q.2

# In[98]:


#Importing PIL for drawing font and image
from PIL import Image, ImageDraw,ImageFont 


# In[112]:


i = Image.new("RGB", (400, 600), "black") #Variety,size and color of image
draw = ImageDraw.Draw(i,"RGBA")
w, h = i.size

space = 30

#creating pattern on the black font
for n in range(space, w, space):
    for x in range(space, h - space, space):
        draw.text((n, x),"£",fill="white", font=ImageFont.truetype("arial"))
i.save("synthetic.tif") #Saving pattern


# In[113]:


img1=cv2.imread('synthetic.tif') #Reading pattern


# In[114]:


#I splitted the pattern image to RGB
img1_B1 = img1[:,:,0]
img1_G1 = img1[:,:,1]
img1_R1 = img1[:,:,2]

#Showing them
cv2.imshow('image_B1',img1_B1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image_G1',img1_G1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('image_R1',img1_R1)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Q.3

# In[102]:


import cv2
import numpy as np

image =cv2.imread('portrait_of_a_young_woman.jpg',0)

#This function which takes an image and a kernel and returns the convolution of them.
def convolve2d(image,kernel):
    
    #Creating zero padd of image
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1,1:-1] = image
    
    #Creating loop for every pixel of the image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            #element-wise multiplication of the kernel and the image
            output[y, x]=(kernel * image_padded[y: y+3, x: x+3]).sum()
    return output

#Kernel to be used Gaussian Blur
KERNEL = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0 #This values about gaussian blur
image_filtering = convolve2d(image, kernel=KERNEL)

#Saving filtered image
cv2.imwrite('filtered_image.jpg', image_sharpen)

#Showing new filtered image 
filtered_image = cv2.imread('filtered_image.jpg')
cv2.imshow('aaa',filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




