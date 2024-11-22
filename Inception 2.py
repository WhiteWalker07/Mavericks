"""
Program: FBD Diagram Analysis with OCR and Shape Detection
Author: [Your Name]
Description:
    This section of the code detects text and shapes in an image, leveraging EasyOCR
    and Tesseract for text recognition, and OpenCV for image processing and contour analysis.
    The goal is to identify objects, their labels, and associated features in the image.
"""

# Importing necessary libraries
import cv2
import pytesseract
import os
import numpy as np
import easyocr as ey

# Set the Tesseract executable path (Update this according to your system installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define a function to calculate the slope between two points
def calculate_slope(point1, point2):
    """
    Calculate the slope of the line connecting two points.

    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        float: Slope of the line.
    """
    return (point2[1] - point1[1]) / (point2[0] - point1[0])


# -------------------------- Image Preprocessing and Setup -------------------------- #

# Define the input image path (Update this with the actual file path)
image_path = r'C:\Users\suyash\Desktop\KACHRA\laohub\Ajgar\FROST HACK\Frost Hack Video\Shape34.png'

# Load the input image
input_image = cv2.imread(image_path)

# Display the input image (for debugging purposes)
cv2.imshow("Input Image", input_image)

# Resize and preprocess the image
processed_image = cv2.resize(input_image, (500, 500), interpolation=cv2.INTER_AREA)
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Define a rectangular kernel for morphological operations
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# -------------------------- Text Detection using EasyOCR -------------------------- #

# Initialize the EasyOCR reader
ocr_reader = ey.Reader(['en'])

# Perform text recognition on the binary image
detected_text = ocr_reader.readtext(binary_image)

# Save recognized text to a file
with open("recognized_text.txt", "a") as text_file:
    recognized_objects = []  # Store recognized text with bounding box details
    for text_data in detected_text:
        recognized_text = text_data[1]
        text_file.write(recognized_text + "\n")
        print(f"Recognized Text: {recognized_text}")

        # Store the text and its bounding box details
        recognized_objects.append([
            int(text_data[0][0][0]),  # Top-left x-coordinate
            int(text_data[0][0][1]),  # Top-left y-coordinate
            int(text_data[0][2][0]) - int(text_data[0][0][0]),  # Width
            int(text_data[0][2][1]) - int(text_data[0][0][1]),  # Height
            recognized_text  # Detected text
        ])

# -------------------------- Fallback Text Detection using Tesseract -------------------------- #

# If EasyOCR does not detect any text, use Tesseract OCR as a fallback
if not recognized_objects:
    for contour in contours:
        # Get the bounding box for the contour
        x, y, width, height = cv2.boundingRect(contour)

        # Draw a rectangle around the detected region (for visualization)
        cv2.rectangle(processed_image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Crop the region of interest for OCR
        cropped_region = binary_image[y:y + height, x:x + width]

        # Recognize text using Tesseract OCR
        tesseract_text = pytesseract.image_to_string(cropped_region)
        
        # Append the detected text to the list
        recognized_objects.append([x, y, width, height, tesseract_text])

# -------------------------- Utility Function for Output Location -------------------------- #

def get_output_location():
    """
    Prompt the user to enter the location to save the output files.

    Returns:
        str: Valid directory path for saving the output.
    """
    location = input("Please enter the location where you want to save the output files: ")
    if os.path.isdir(location):
        return location
    else:
        print("Invalid location. Please try again.")
        return get_output_location()



blank = np.zeros(img.shape, dtype='uint8')

# Resizing the image 
hgt = img1.shape[0]
wdt = img1.shape[1]

if hgt>500 & wdt>500:
    img = cv2.resize(img1, (700,700), interpolation = cv2.INTER_AREA)

else:
    img = cv2.resize(img1, (700,700), interpolation=cv2.INTER_CUBIC) 

blank = np.zeros(img.shape, dtype='uint8')          
eli = []


# converting image into grayscale image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
arr = np.array(gray)

#To find the average brigthness in a image
"""
avg = np.sum(arr, dtype= np.int32)//490000
print(avg)
"""


# setting threshold of gray image
_, threshold = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
#threshold = cv2.erode(threshold1, (7,7), iterations=4)
#cv2.imshow("hello", threshold)                                       #Here many hit and trials where done to find the best formate for furthur procedure

# using a findContours() function
contours, _ = cv2.findContours(
	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blank, contours, -1, (0,0,255), 1)
#cv2.imshow("draw", blank)


# This part of the program is responsible for detection of all the shapes and arrows whichh are detected

i = 0
sumt = []
liobj =[]

for contour in contours:
    sumx = 0
    sumy = 0
    minx = 1000
    miny = 1000
    maxx = 0
    maxy = 0
    minimum = []
    maximum = []
    

    # This sub part is detection the shape by first unifying contours into shapes 
    Area = cv2.contourArea(contour)
    if Area > 5000:
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
        liobj.append([x,y])   
        
        xc, yc, wc, hc = cv2.boundingRect(contour)
        lianother.append([xc,yc,xc+wc,yc+hc,int(0)])

        #cv2.imshow('arrow', img) 
        eli.append(Area)
        
        # putting shape name at center of each shape and drawContours() function
        if len(approx) == 3:
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
            cv2.putText(img, 'Triangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(approx) == 4:
            cv2.drawContours(img, [contour], 0, (0,255, 255), 1)
            cv2.putText(img, 'Quadrilateral', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255, 255), 2)           
        elif len(approx) == 5:
            cv2.drawContours(img, [contour], 0, (0, 255,0), 1)
            cv2.putText(img, 'Pentagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255),2)
        elif len(approx) == 6:
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 1)
            cv2.putText(img, 'Hexagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif len(approx) == 8:
            cv2.drawContours(img, [contour],0, (255,0,0), 1)
            cv2.putText(img, 'Octagon', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0,0 ), 2)    
        else:
            cv2.drawContours(img, [contour], 0, ( 255, 255,0), 1)
            cv2.putText(img, 'circle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 255, 255,0), 2)


    # ARROW IDENTIFICATION

    # This code down below os for arrow identification so to identif a arrow we are assigning it with a area limit 
    # and the minimum number of side limit to eliminate athe noise. The to identify the direction of the force we 
    # are finding the centroid of the arrow and then finding the minimum and maximum x,y to find the centre and then 
    # as we know that points of arrow ar concentrated at the arrow side so the line between the centre and the centroid 
    # is the direction and then we draw a line at the x coordinate to find the angle by equading the slopes. 
    # Then to identify the coordrant of the angle we equaded the location of the centroid and the centre 
    # Drawback of this logic is that I think it will final to identift the direction in case of flat arrow as the 
    # concentrated point are close to the tail point it may be that the cetre gets to the opposite side of the centroid 
    # as if we take a special case of arrow which is a triangle then the centre will be ahead of the centroid 


    elif 1000<Area<5000:
                app = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
                #cv2.drawContours(img, [app], 0, (255, 30, 50), 1)              #contours


                if 6<=len(app)<=10 :
                        for i in range(len(app)):

                                ex = app.ravel()[int(2*i)]
                                ey = app.ravel()[int(2*i+1)]
                                sumx = sumx + ex
                                sumy = sumy + ey
                                if minx>ex:
                                        minx =ex
                                if miny>ey:
                                        miny = ey
                                if maxx<ex:
                                        maxx = ex
                                if maxy<ey:
                                        maxy = ey  
                                
                                #wha = len(app)
                                #print(app.ravel(),wha)
                                #cv2.circle(img, (x,y), 10, (0, 0,255))
                                #cv2.circle(img, (ex,ey), 2, (10, 10,255),-1)                   #Points
                                side = int(len(app) - 1)
                                if i==side :
                                        
                                        cv2.circle(img,(sumx//(side+1),sumy//(side+1)) , 3, (0,255,0), thickness = -1)
                                        cv2.circle(img, ((minx+maxx)//2,(miny + maxy)//2) , 3, (0,255,0),thickness = -1) 
                                        cv2.circle(img, ((((maxx + minx)//2)+40),(miny + maxy)//2) , 3, (0,255,0),thickness = -1) 
                                        minimum.append([minx,miny])
                                        maximum.append([maxx,maxy])
                                        cv2.rectangle(img, (minx-15,miny-15), (maxx+15,maxy+15), (10,40,80), thickness = 1)
                                        cv2.line(img, ((minx+maxx)//2,(miny + maxy)//2),(sumx//(side+1),sumy//(side+1)), (255,120,30))
                                        cv2.line(img, ((minx+maxx)//2,(miny + maxy)//2),((((maxx + minx)//2)+40),(miny + maxy)//2), (255,120,30))
                                        p2 = ((minx+maxx)//2,(miny + maxy)//2)
                                        p3 = (sumx//(side+1),sumy//(side+1))
                                        p1 = ((((maxx + minx)//2)+40),(miny + maxy)//2)
                                        m1 = slope(p1,p2)
                                        m2 = slope(p3,p2)
                                        
                                        ang = (m1 - m2)/(1 + m1*m2)
                                        angle = math.atan(ang)
                                        angle = math.degrees(angle)
                                        

                                        if p2[0]>=p3[0] :
                                                if  p2[1]>=p3[1]:
                                                        cv2.putText(img, "Angle="+str(180 + angle)[:5], (minx-15,miny-20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (120,120,255))
                                                        cen = (maxx,maxy)
                                                        count = (minx, miny)
                                                        plus = (10,10)
                                                        sumt.append([sumx//(side+1),sumy//(side+1),minx,miny,maxx,maxy, float(str(180 + angle)[:6])])

                                                else: 
                                                        cv2.putText(img, "Angle="+str(180 + angle)[:6], (minx-15,miny-20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (120,120,255))
                                                        cen = (maxx, miny)
                                                        count = (minx, maxy)
                                                        plus = (10, -10)
                                                        sumt.append([sumx//(side+1),sumy//(side+1),minx,miny,maxx,maxy, float(str(180 + angle)[:6])])
                                        else :
                                                if p2[1]>=p3[1]:
                                                        cv2.putText(img,"Angle="+ str(angle)[:5], (minx-15,miny-20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (120,120,250))
                                                        cen = (minx, maxy)
                                                        count = (maxx,miny)
                                                        plus = (-10,10)
                                                        sumt.append([sumx//(side+1),sumy//(side+1),minx,miny,maxx,maxy, float(str(angle)[:5])])

                                                else: 
                                                        cv2.putText(img,"Angle="+ str(270 - angle)[:6], (minx-15,miny-20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (100,100,255))
                                                        cen = (minx,miny)
                                                        count = (maxx, maxy)
                                                        plus = (-10,-10)
                                                        sumt.append([sumx//(side+1),sumy//(side+1),minx,miny,maxx,maxy, float(str(270 -angle)[:6])])
                                        
                                        cv2.line(img, cen, (cen[0], count[1]), (255,0,255), thickness = 1)   
                                        cv2.line(img, cen, (count[0], cen[1]), (255,0,255), thickness = 1) 
                                        cv2.line(img, (count[0], cen[1]), (count[0] + plus[0], cen[1] + 10), (255,0,255), thickness = 1)
                                        cv2.line(img, (count[0], cen[1]), (count[0] + plus[0], cen[1] - 10 ), (255,0,255), thickness = 1)
                                        cv2.line(img, (cen[0], count[1]), (cen[0] - 10, count[1] + plus[1]), (255,0,255), thickness = 1)
                                        cv2.line(img, (cen[0], count[1]), (cen[0] + 10, count[1] + plus[1]), (255,0,255), thickness = 1)



                        cv2.circle(img, (minx,miny), 2, (10, 10,255),-1)                        
                        cv2.putText(img,"Arrow", (minx,miny -35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))


cv2.imshow('shapes', img)

# ELIMINATION FO EXTRA CENTRES BEING DETECTED

# In this logic first we are eliminating the outer and inner edge error by taking out the distance between the centres 
# as if they are clearing the area cutoff then two body should be at a min distance therefore using this interpretation 
# we are elimination the overlap error and the comes the oct-rect error to  emove this we are taking out the number of centre 
# in the area of the body and if they are greater than or equal to 3 then we remove it 3 was taken as in octa-rect error one of 
# rectangles will have the centre of octagon so to prevent that rectangle to get eliminated.


for i in range(len(liobj)):
    w=0
    for x in range(len(liobj)-i):
        if (-10<liobj[i][0] -liobj[x+i-w][0] < 10) & (-10<liobj[i][1] - liobj[x+i-w][1]<10) & (x+i-w != i):
            liobj.remove(liobj[x+i-w])
            lianother.remove(lianother[x+i-w])
            eli.remove(eli[x+i-w])
            w= w+1

    ec = 0
    for x in range(len(liobj)):
        s = 0
        for cen in liobj:
            if (lianother[x-ec][0] < cen[0] < lianother[x-ec][0] + lianother[x-ec][2]) &  (lianother[x-ec][1] < cen[1] < lianother[x-ec][1] + lianother[x-ec][3]):
                s = s + 1
        if s >=3 :
            liobj.remove(liobj[x-ec])
            lianother.remove(lianother[x-ec])
            ec = ec +1
            eli.remove(eli[x-ec])


"""
for i in range(len(text)):
    w=0
    for x in range(len(text)-i):
        if (-10<text[i][0] -text[x+i-w][0] < 10) & (-10<text[i][1] - text[x+i-w][1]<10) & (x+i-w != i):            
            text.remove(text[x+i-w])            
            w= w+1
"""
# This logic is working and is to eliminate the octa-rect error            
"""
for l in liobj:
    x = l[0]
    y = l[1]           
    img = cv2.line(img, (x,y), (x, y+int(h/1.5)) , (255,0,0), 3)
    img = cv2.line(img, (x,y+int(h/1.5)), (x-int(x/10),y+int(h/1.8)),(255,0,0), 3)
    img = cv2.line(img, (x,y+int(h/1.5)), (x+int(x/10),y+int(h/1.8)) ,(255,0,0),3)
    img = cv2.putText(img, "mg", (x-20,y+int(h/1.5)+20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
    img = cv2.line(img, (x,y), (x, y-int(h/1.5)) , (0,0,255), 3)
    img = cv2.line(img, (x,y-int(h/1.5)), (x+int(x/10),y-int(h/1.8)),(0,0,255), 3)
    img = cv2.line(img, (x,y-int(h/1.5)), (x-int(x/10),y-int(h/1.8)) ,(0,0,255),3)
    img = cv2.putText(img, "N", (x-20,y-int(h/1.5)-20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)
"""

# Elimating noise in text recogination

w = 0
for i in range(len(text)):
    if text[i-w][4] == '\x0c' or text[i-w][4] == '\n\x0c' or text[i-w][4] == '':       #This noise removal code was fro tesseract
        text.remove(text[i-w])
        w = w+1

#Combining the two list of centres and edges
liobj = np.array(liobj)
liobj = liobj.reshape(int(len(liobj)),2)

lianother = np.array(lianother)
lianother = lianother.reshape(int(len(lianother)), 5)
#print(liobj, lianother)                                               #This is to find problem just before the merging of arrays
shape =  np.concatenate((liobj, lianother),axis = 1)
shape = shape.tolist()


# This is the part making arrows for the program and writing mg and N. The values are writen in the next part
for i in shape:
    x,y,_,_,_,ym,_ = i
    hc = ym - y
    img = cv2.line(img, (x,y), (x, y+int(hc/1.5)) , (255,0,0), 3)
    img = cv2.line(img, (x,y+int(hc/1.5)), (x-int(x/10),y+int(hc/1.8)),(255,0,0), 3)
    img = cv2.line(img, (x,y+int(hc/1.5)), (x+int(x/10),y+int(hc/1.8)) ,(255,0,0),3)
    img = cv2.putText(img, "mg", (x-20,y+int(hc/1.5)+20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
    img = cv2.line(img, (x,y), (x, y-int(hc/1.5)) , (0,0,255), 3)
    img = cv2.line(img, (x,y-int(hc/1.5)), (x+int(x/10),y-int(hc/1.8)),(0,0,255), 3)
    img = cv2.line(img, (x,y-int(hc/1.5)), (x-int(x/10),y-int(hc/1.8)) ,(0,0,255),3)
    img = cv2.putText(img, "N", (x-20,y-int(hc/1.5)-10) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)


#Finding where are different variables

# Here we are taking out the the target string and the number with it by usiing re library target->to uppercase->finding ->storing
kilo = []
nor = []
e = 0
vari = []

for x in text:

    #Finding the location of the variable
    loc = x[4].upper().find("KG")
    locf = x[4].find("N")
    
    #locating the closest number to the string
    numd = x[4][:int(loc)]
    numf = x[4][:int(locf)]
    
    #Making the string pattern we want
    reco = re.compile('\D')
    
    #Applying the string pattern to be found
    n = reco.split(numd)
    f = reco.split(numf)
    
    # Making the variables storing the values null to avoid unnecessary values
    nkg = []
    nn = []
    for i in n:
        if i != '' and loc!= -1:
            nkg = i

    if nkg != []:       
        kilo.append([int((x[0]+ x[2]/2)*1.4), int((x[1] + x[3]/2)*1.4) ,int(nkg),"kg"]) 

    for i in f:
        if i != '' and locf!= -1:
            nn = i 

    if nn!=[]:
        nor.append([int((x[0]+ x[2]/2)*1.4), int((x[1] + x[3]/2)*1.4),int(nn),"N"])

    if nkg == [] and nn == []:
        vari.append(e)  
    e = e+1
variable = []
for i in vari:
    variable.append([int((text[i][0]+ text[i][2]/2)*1.4), int((text[i][1] + text[i][3]/2)*1.4),text[i][4],"vari"])

print(variable)

#print(liobj)                   #Centres
#print(lianother)               #Edegs
print(shape)                    
#print(eli)                     #Area
#print(text)                    #text seen by the program
print(sumt)
net = kilo + nor + variable
print(net)            


#Phase 2 Class and unification of variables

combo = shape + sumt
#print(combo)                           #Combined forces and mass arrays
for x in combo:
    minxd = 10000
    w = -1
    
    s = None
    for i in net:
        xc = x[0] - i[0]
        yc = x[1] - i[1]
        dist = math.sqrt(xc*xc + yc*yc)
        w= w + 1
        if dist<minxd:
            minxd = dist
            s = w
    
    x.insert(-1,net[s][2])
    x.insert(-1,net[s][3])
        


class fbd():

    def __init__(self ,mass):
        self.mass = mass
        #self.angle = angle
            
print(combo)

final =[]
for i in range(len(combo)):
    if combo[i][-2] == "kg":
        final.append(combo[i])
    else:
        final.append(combo[i])


#PREVIOUS TRYS

# Here are two logics which are commented in the first one i have a doubt that when i am giving two values to unpack 
# but still a error is being shown . In the second part my approch was to target the object and find the nearest force
# in this method the drawbacks are that if their is only one force present and twobodies are present then both of the 
# bodies will claim that it is a force for that body and if forces are more than the number of objects then some forces 
# will be left out. So heres the point froce can be applied on only one body but bosy can have multiple forces. 
# So the third logic which is currently active working one find the object closest to the force and then assigning the 
# value of force to the body
"""
for  _,i in final:    
    num = 0
    w = 0
    z=0
    for x, _ in final:
        if i != None:
            if x != None:
                                
                xf = (i[2] - x[0])
                yf = (i[3] - x[1])
                xb = (i[4] - x[0])
                yb = (i[5] - x[1])

                distf = math.sqrt(xf*xf + yf*yf)
                distb = math.sqrt(xb*xb + yb*yb)

                if distf>=distb:
                    dist = distf
                else:
                    dist = distb                                       #This part is not okay
                
                if z == -1 :
                    if mindist > dist:
                        mindist = dist
                        num = w
                        
                else:
                    mindist = dist
                    num = 0
                    z = -1
                    
                w = w+1
        
        if w == len(final)-2:
            print(mindist)
            final[num].append(i[-3])
            final[num].append(i[-2])
            final[num].append(i[-1])
            
"""
"""
for x,_ in final:                           #Here x is mass and y is angle
    w=0
    if x != None:
        for _ , i in final:
                        
                if i != None:
                    xf = (x[0] - i[2])
                    yf = (x[1] - i[3])
                    xb = (x[0] - i[4])
                    yb = (x[1] - i[5])
                    distf = math.sqrt(xf*xf + yf*yf)
                    distb = math.sqrt(xb*xb + yb*yb)                         #This part is also ok
                    if distf >= distb:
                        dist = distf
                    else :
                        dist = distb
                    if w != 0:
                        w = -1
                        if mindist > dist:
                            mindist = dist
                            arr = [i[-3],i[-2],i[-1]]
                            
                            
                    else:
                        mindist = dist
                        w = -1
                        arr = [i[-3],i[-2],i[-1]]
        
        if w == -1:              
            x.append(arr[0])
            x.append(arr[1])
            x.append(arr[2])
            

print()

killcount = []
for num , (trash,_) in enumerate(final):                
    if trash == None:
        killcount.append(num)

w=0
for num in killcount:
    final.remove(final[num -w])                                  #This part is ok 
    w = w+1
""" 

print(final)       


# Storing both mass and angle in one variable the current successor logic of the above logics
remove = []
for c,i in enumerate(final):
    if i[7] == "N" or i[7] == "vari":
        remove.append(c)
        w = 0
        num = 0
        count = 0

        for  x in final:
            count = count + 1

            if x[7] == "kg":
                xf = (x[0] - i[2])
                yf = (x[1] - i[3])
                xb = (x[0] - i[4])
                yb = (x[1] - i[5])

                distf = math.sqrt(xf*xf + yf*yf)
                distb = math.sqrt(xb*xb + yb*yb)

                if distf <= distb:
                    dist = distf
                else :
                    dist = distb
                
                if w!=0:
                    w = w-1
                    if mindist > dist:
                        mindist = dist
                        num  = count -1 
                
                else:
                    w = w-1
                    mindist = dist
                    num =0
            
            if count == len(final) - 1:
                #print(num)
                #print(mindist)                                #This was used to detect the big > and < blunder in distance comparision         
                final[num].append(i[-3])
                final[num].append(i[-2])
                final[num].append(i[-1])                
#print(remove)                                                 

#This print statement is that what values are to be removed from array
for x in range(len(remove)):
    final.remove(final[remove[x] - x])
print("The final compressed form to present data extracted form the image about the bodies:\n",final)


#This is the program to find the components of forces in appied on each body
finalcom = []
unknowns = []

c=0
for x in final:
    y = int(len(x))
    if y >=10:
        eqx = 0 
        eqy = 0 
        for i in range(int((y-9)/3)):
            if x[10+(i*3)] == "N":
                xcom = (x[9+(i*3)])*(math.cos(math.radians(x[11+(i*3)])))
                ycom = (x[9+(i*3)])*(math.sin(math.radians(x[11+(i*3)])))
                eqx = eqx + xcom
                eqy = eqy + ycom
            elif x[10+(i*3)] == "vari":
                unknowns.append([c,x[9+(i*3)],x[11+(i*3)]])

        c = c+1
        finalcom.append([eqx,eqy])

print("components of the forces applied : ",finalcom)
print("variables:",unknowns)


#This is a fuction desizned extract data related to the variables from the user. This is not active and is under proposal phase.
def extraction():
    minimum = []
    ANG = [0, 30, 45, 60, 90, 120, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]
    for un in unknowns:
        for a in ANG:
            minimum.append(un[2] - a)
        print(ANG[minimum.index(min(minimum))])  

    for x in unknowns:
        s = input("Information related to",x[1],"this Variable\nPress 1 if the variable angle is given\nPress 2 if the variable has no information given\nPress 3 if its a value with both angle and value known\nPress 4 if their is no such variable in the question\n\nPlease enter the most suitable option : ")
        if s==3:
            val = input("Please enter the value of",x[1],"with unit")
            ang = input("Please enter the angle at which",x[1],"force is pointing")
        elif s==1:
            ang = input("Please enter the angle at which",x[1],"force is pointing")
        elif s == 2:
            pass
        elif s == 4:
            pass
        else:
            extraction()


#SURFACE IDENTIFICATION

#The surfaces of the object will be their low and upper side that means if a object is above another object
#then the centre of the above object should be located between (minimum x of the lower object,0) and 
# (maximum of x of the rectangle, minimum of y of the rectangle)

normal = []
friction = []
i= 0

for up in final:
    nor = up[6]*10 - finalcom[i][1]
    j=-1
    frict = finalcom[i][0]
    for down in final:
        j = j+1
        if (up[2]<down[0]<up[4]) and (0<down[1]<up[3]) and i != j:
            nor = nor + down[6]*10 - finalcom[j][1]
            frict = frict + finalcom[j][0]
    i = i+ 1
    normal.append(float(str(nor)[:5]))

    if frict>0:
        friction.append(float(str(frict)[:5]))
    else:
        friction.append(float(str(frict)[:6]))
print("normal : ",normal)
print("friction force required to keep the system static : ",friction)


#Display of values of normal and weight 
c=0
for i in shape:
    x = i[0]
    img = cv2.putText(img, "   ="+str(i[6]*10)+" N", (i[0]-20,i[1]+int(i[5]-i[3])//2-10) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
    img = cv2.putText(img, "  ="+str(normal[c])+" N", (i[0]-20,i[1]-int(i[5]-i[3])//2+20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)
    c=c+1


#Logic not working 
"""
li1 = np.array(lianother)
litomany = li1.flatten()
liop = len(litomany)/4
linew = []
for q in range(int(liop)):
    for var in range(int(liop)):

        if lianother[q][0]<=lianother[var][0] & lianother[q][1]<=lianother[var][1] & var != i  &  (lianother[q][0]+lianother[q][2])<(lianother[var][0]+lianother[var][2]) &  (lianother[q][1]+lianother[q][3])<(lianother[var][1]+lianother[var][3]):
            lianother.remove(lianother[var])
""
# Logic not working

#Method 2 of storing and displaying data

#print(lianother)
f = open("recognized.txt", "r")
read = f.read()
read = read.replace(" ", "")
read = read.replace("\n", "")
read = read.replace("", "")
#print(read)
f.close()

digit = re.compile('\D')
n = digit.split(read)
num = []
for i in n:
    if i != '':
        num.append(i)
        
digit = re.compile('\d')
a = digit.split(read)
alpha = []
for i in a:
    if i != '':
        alpha.append(i)
print(num)        
print(alpha)

kg = []
keyset = []
for i in alpha:
    if i.upper() == 'KG':
        kg.append(i)
if len(liobj) == 1:
    keyset.append(str(liobj[0])+";" + str(num[0])) 
elif len(liobj) == 2:
    if len(kg) == 1:
        if liobj[0][1]>liobj[1][1]:
            keyset.append(str(liobj[1])+";"+str(num[0]))
        else:
            keyset.append(str(liobj[0])+";"+str(num[0]))
    elif len(kg) == 2:
        if liobj[0][1]>liobj[1][1]:
            keyset.append(str(liobj[1])+";"+str(num[0]))
            keyset.append(str(liobj[0])+";" + str(num[0]))
        else:
            keyset.append(str(liobj[0])+";"+str(num[1]))
            keyset.append(str(liobj[1])+";" + str(num[1]))  
else:
    pass

if len(keyset) == 1:
    s = keyset[0]
    key = s.split(";")
    img = cv2.putText(img, "   ="+str(int(key[1])*10)+" N", (liobj[0][0]-20,liobj[0][1]+int(h/1.5)+20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
    img = cv2.putText(img, "  ="+str(int(key[1])*10)+" N", (liobj[0][0]-20,liobj[0][1]-int(h/1.5)-20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)
    img = cv2.putText(img, "N=mg="+str(int(key[1])*10)+" N", (10,25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (30, 120, 255), 1)


elif len(keyset) == 2:
    nor = 0
    l=-1

    for i in range(2):
        l +=1
        kyu = keyset[i]
        #print(kyu)
        key = kyu.split(";")
        x =  key[0]
        nor += int(key[1])
        st = re.compile('\D')
        dig = st.split(x)
        dhinchak = []
        for i in dig:
            if i != '':
                dhinchak.append(i)
        x = dhinchak[0]
        y =dhinchak[1]
        
        if l==0:
            img = cv2.putText(img,"BODY1: n=mg="+str(int(nor)*10)+" N", (10,25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (30, 120, 255), 1)
        else:
            img = cv2.putText(img,"BODY2: Mg+mg=N="+str(int(nor)*10)+" N", (10,675), cv2.FONT_HERSHEY_COMPLEX, 0.5, (30, 120, 255), 1)
        img = cv2.putText(img, "   ="+str(int(key[1])*10)+" N", (int(x)-20,int(y)+int(h/1.5)+20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
        img = cv2.putText(img, "   ="+str(int(nor)*10)+" N", (int(x)-20,int(y)-int(h/1.5)-20) ,cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 1)
"""


#For finding the cofficient of restitution on the body
i = 0 
cof = []
for  x in normal:
    coffof = friction[i]/x
    i = i+1
    cof.append(abs(float(str(coffof)[:5])))

print("Cofficient of friction on the respective bodies should be :",cof) 


# This part is writing the cofficient of restitution and the frictional force needed to keep the body in rest
i = 0 
for x in friction:
    if x>0:
        img = cv2.line(img, (final[i][2], final[i][5]),(final[i][2] - 60, final[i][5]),(120,0,180),thickness = 2 )
        img = cv2.line(img, (final[i][2] - 60, final[i][5]), (final[i][2] - 50, final[i][5]+10),(120,0,180),thickness = 2 )
        img = cv2.line(img, (final[i][2] - 60, final[i][5]), (final[i][2] - 50, final[i][5]-10),(120,0,180),thickness = 2 )
        img = cv2.putText(img,str(x)+" N", (final[i][2]-130, final[i][5]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
        img = cv2.putText(img,"u="+str(cof[i]), (final[i][2]-100, final[i][1]),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    elif x==0:
        pass
    else:
        img = cv2.line(img, (final[i][4], final[i][5]),(final[i][4] + 100, final[i][5]),(120,0,180),thickness = 2 )
        img = cv2.line(img, (final[i][4] + 100, final[i][5]), (final[i][4]+ 90, final[i][5]+10),(120,0,180),thickness = 2 )
        img = cv2.line(img, (final[i][4] + 100, final[i][5]), (final[i][4] + 90, final[i][5]-10),(120,0,180),thickness = 2 )
        img = cv2.putText(img,str(abs(x)) + " N", (final[i][4]+110, final[i][5]),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1) 
        img = cv2.putText(img,"u="+str(cof[i]), (final[i][4]+150, final[i][1]),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1) 
    i = i + 1

# Asking if the user wants to save the image
choice = input("Do you want to save the FBD made on your device(Y/N)")

OLDDIR = os.getcwd()
OLDDIR = OLDDIR.replace("\\","/")

if choice == "Y" or choice == 'y':
    filename = input("Please enter the folder name")
    NEWDIR = locat()
    NEWDIR = NEWDIR.replace("\\","/")

    os.chdir(NEWDIR)
    os.mkdir(filename)
    
    os.chdir(NEWDIR+r"/"+filename)
    cv2.imwrite("OUTPUT IMG.png", img)
    
    f = open("OUTPUT TXT.txt", 'w')

    for i in range(len(finalcom)):
        INTRO = "The analysis done on the first figure :-\n"
        XCOMP = str("\nThe equivalent X-component is " + str(finalcom[i][0])[:5])
        YCOMP = str("\nThe equivalent Y-component is " +  str(finalcom[i][1])[:5])
        NORML = str("\nThe Normal Reaction is "+ str(normal[i]))
        FCOFF = str("\nThe minimum cofficient of friction to keep the system static: " + str(cof[i])+"\n\n")
        
        f.write(INTRO)
        f.write(XCOMP)
        f.write(YCOMP)
        f.write(NORML)
        f.write(FCOFF)

if choice == 'Y' or choice == 'y':
    f.close()

# The Standard end statement of a program
os.chdir(OLDDIR)
os.remove("recognized.txt")
cv2.imshow("end", img)
cv2.waitKey(0)
cv2.destroyAllWindows()