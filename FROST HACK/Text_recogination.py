import cv2
import pytesseract
import os
import numpy as np
import re

# Configure Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'...\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = "Shape13.jpg"
original_image = cv2.imread(image_path)

# Resize the image for uniform processing
resized_image = cv2.resize(original_image, (500, 500), interpolation=cv2.INTER_AREA)

# Convert the resized image to grayscale for better processing
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Extract text using Tesseract OCR from the grayscale image
detected_text = pytesseract.image_to_string(grayscale_image)

# Threshold the grayscale image for contour detection
_, binary_threshold = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Define a rectangular kernel for morphological operations
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

# Find contours from the binary image
contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Prepare variables to store detected object centers and shapes
detected_centers = []

# Process each contour for shape and center detection
for contour in contours:
    # Get the bounding box for the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Draw the bounding box on the image
    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the detected region for text recognition
    cropped_region = resized_image[y:y + h, x:x + w]

    # Save recognized text to a file
    with open("recognized.txt", "a") as file:
        recognized_text = pytesseract.image_to_string(cropped_region)
        file.write(recognized_text)

# Convert the image back to grayscale
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Further thresholding for shape detection
_, threshold_image = cv2.threshold(grayscale_image, 120, 255, cv2.THRESH_BINARY)

# Find contours again after thresholding
contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours to identify shapes
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 2000:  # Ignore small areas
        # Approximate the contour to reduce vertices
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

        # Calculate the center of the shape using moments
        moments = cv2.moments(contour)
        if moments['m00'] != 0.0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
            detected_centers.append([center_x, center_y])

        # Draw arrows for forces acting on the detected shape
        img = cv2.line(resized_image, (center_x, center_y), (center_x, center_y + 50), (255, 0, 0), 3)  # mg (weight)
        img = cv2.line(resized_image, (center_x, center_y), (center_x, center_y - 50), (0, 0, 255), 3)  # N (normal)

        # Annotate the shape based on the number of vertices
        if len(approx) == 3:
            shape_name = "Triangle"
        elif len(approx) == 4:
            shape_name = "Quadrilateral"
        elif len(approx) == 5:
            shape_name = "Pentagon"
        elif len(approx) == 6:
            shape_name = "Hexagon"
        elif len(approx) == 8:
            shape_name = "Octagon"
        else:
            shape_name = "Circle"

        # Display the name of the shape on the image
        cv2.putText(resized_image, shape_name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Load text file and process recognized data for further use
with open("recognized.txt", "r") as file:
    raw_text = file.read().replace(" ", "").replace("\n", "").replace("\x0c", "")

# Extract numeric and alphabetic values from the text
numbers = [int(num) for num in re.split(r'\D+', raw_text) if num]
letters = [char for char in re.split(r'\d+', raw_text) if char]

# Process the detected objects and associate forces
kg_objects = []
force_objects = []
for letter in letters:
    if letter.upper() == 'KG':
        kg_objects.append(letter)

# Annotate forces and results on the image
if len(detected_centers) == 1 and numbers:
    center = detected_centers[0]
    weight = numbers[0] * 10
    cv2.putText(resized_image, f"Weight={weight}N", (center[0] - 20, center[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)

elif len(detected_centers) == 2 and len(numbers) >= 2:
    for i, center in enumerate(detected_centers):
        weight = numbers[i] * 10
        cv2.putText(resized_image, f"Weight={weight}N", (center[0] - 20, center[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)

# Display the final processed image
cv2.imshow("Final Image with Shapes and Forces", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Clean up temporary files
os.remove("recognized.txt")
