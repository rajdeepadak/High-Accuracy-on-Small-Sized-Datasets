import cv2 
import os
import numpy as np



def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



for subdir, dirs, files in os.walk(r'E:\Learning\BARC Work\Determine_Illumination_Characteristics\NL'):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".jpg"):
                img = cv2.imread(filepath)
                img = cv2.resize(img,(480, 360))
                
                g_img = adjust_gamma(img, 0.1)
                
                hsv_g_img = cv2.cvtColor(g_img, cv2.COLOR_BGR2HSV)
                
                H, S, V = cv2.split(hsv_g_img)
                
                cv2.imshow('V', V)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                result = hsv_g_img.copy()
                
                lower = np.array([0, 0, 0])
                upper = np.array([1, 0, 255])
                
                mask = cv2.inRange(hsv_g_img, lower, upper)
                
                result = cv2.bitwise_and(result, result, mask=mask)                
                
                cv2.imshow('HSV Image', hsv_g_img)
                cv2.imshow('Light Sources', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                


