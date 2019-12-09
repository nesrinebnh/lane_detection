import cv2
import numpy as np
import matplotlib.pyplot as plt
##edge detetction algorithm the pricnipe of this algorithm is to find the point where the colors changed
##and edge is defined by the difference in intensity values in adjacent pixels 
##and wherever there's a sharp change in intensity
##a rapid change in brightness wherever there's a strong 
##gradient there is a corresponding bright pixel in the gradient image by
##tracing out all these pixels we obtain the edge

##to detect lanes in image we follow these steps
##step1 we convert our image to grayscale
##because the normal color had three channe blue green and red
##or the gray has only one channel varies between 0 and 255

def canny(image):
	##cvtColor convert an image from one color to another
	gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

	##step 2 apply Gaussian blur on image to reduce noise
	blur = cv2.GaussianBlur(gray, (5,5), 0)

	##step3 finding region of interest lane lines
	canny = cv2.Canny(blur, 50,150)

	return canny

def display_lines(image,lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			print(line)
			#we have 2D with 4column
			x1,y1,x2,y2 = line.reshape(4)
			cv2.line(line_image,(x1,y1), (x2,y2), (255,0,0), 10)
	return line_image
			
def region_of_interest(image):
	height = image.shape[0]
	polygons = np.array([
		[(-100,height),(200, height),(430,300)],
		[(200,height),(490, height),(490,270)],
		[(400,height),(750,height),(530,290)],
		[(540,height),(1200,height),(586,300)]
	])
	mask = np.zeros_like(image)
	##triangle will be completely while
	cv2.fillPoly(mask, polygons,255)
	masked_image = cv2.bitwise_and(image, mask)#&

	return masked_image;

##read images
image = cv2.imread('test.jpg')
##lane_image will take a copy any changes on lane_copy won't affect on image
lane_image = np.copy(image)
canny = canny(lane_image)

cropped_image = region_of_interest(canny)
##one point is a line in Hough space
## but in the case of vertical line it's not absolutely the same
##the we will make equations of cos and sin
##2pixel and 1 dgree precision in this case
## maxLineGap is the distance between lines
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 5, np.array([]), minLineLength=2, maxLineGap=2)
line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image,0.8,line_image, 1, 1)
## this will help to identify the area you should calculate it estimate it by tracing a rectangle

#plt.imshow(canny)
#plt.show()
##imshow takes the name of the window as the first parameter and the image as the second
cv2.imshow('result',combo_image)
##display the image for a specified amount of milliseconds 
##it will continue displaying the image until
## we press anything in the keyboard
cv2.waitKey(0) 