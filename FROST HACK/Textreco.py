import cv2
import pytesseract

import numpy as np
from matplotlib import pyplot as plt
  
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
link = r'C:\Users\suyash\Desktop\Shape6.jpeg'
img1 = cv2.imread(link)
img = cv2.resize(img1,(500,500),interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

text = pytesseract.image_to_string(gray)  
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)


for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
      
    # Drawing a rectangle on copied image
    rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
      
    # Cropping the text block for giving input to OCR
    cropped = img[y:y + h, x:x + w]
      
    # Open the file in append mode
    file = open("recognized.txt", "a")
   
    
    # Apply OCR on the cropped image
    text1 = pytesseract.image_to_string(cropped)
    file.write(text)
    file.write("\n")
    file.close()
print(text)
print(text1)
blank = np.zeros(img.shape, dtype='uint8')
# converting image into grayscale image



hgt = img1.shape[0]
wdt = img1.shape[1]
if hgt>500 & wdt>500:
    img = cv2.resize(img1, (500,500), interpolation = cv2.INTER_AREA)
elif hgt<500 & wdt<500:
    img = cv2.resize(img1, (500,500), interpolation=cv2.INTER_CUBIC) 
elif hgt>500 & wdt<500:
     img = cv2.resize(img1, (500,500), interpolation = cv2.INTER_AREA)
     img = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
     pass 
elif hgt<500 & wdt>500:
     img = cv2.resize(img1, (500,500), interpolation = cv2.INTER_AREA)
     img = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
     pass      
blank = np.zeros(img.shape, dtype='uint8')          


# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
_, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
#threshold = cv2.erode(threshold1, (7,7), iterations=4)
cv2.imshow("hello", threshold)

# using a findContours() function
contours, _ = cv2.findContours(
	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blank, contours, -1, (0,0,255), 1)
cv2.imshow("draw", blank)
i = 0

# list for storing names of shapes
for contour in contours:
    
    Area = cv2.contourArea(contour)
    if Area > 2000:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1                #Ye part samj nahi aya
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        
        # finding center point of shape
        M = cv2.moments(contour)
        if M['m00'] != 0.0:
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])

        # putting shape name at center of each shape and drawContours() function
        if len(approx) == 3:
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
            cv2.putText(img, 'Triangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(approx) == 4:
            cv2.drawContours(img, [contour], 0, (0,255, 255), 1)
            cv2.putText(img, 'Quadrilateral', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 255), 2)
        elif len(approx) == 5:
            cv2.drawContours(img, [contour], 0, (0, 255,0), 1)
            cv2.putText(img, 'Pentagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0),2)
        elif len(approx) == 6:
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
            cv2.putText(img, 'Hexagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(approx) == 8:
            cv2.drawContours(img, [contour],0, (255,0,0), 1)
            cv2.putText(img, 'Octagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0 ), 2)    
        else:
            cv2.drawContours(img, [contour], 0, ( 255, 255,0), 1)
            cv2.putText(img, 'circle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 255, 255,0), 2)
# displaying the image after drawing contours
cv2.imshow('shapes', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
