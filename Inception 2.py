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
import math

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



"""
Program Section: Shape Detection and Labeling
Author: [Your Name]
Description:
    This section detects shapes in an image using OpenCV contours, calculates their properties (area, center),
    and labels them based on the number of sides. The results are visualized by drawing contours and annotating shapes.
"""

# Initialize a blank image for visualizing contours
contour_overlay = np.zeros(processed_image.shape, dtype='uint8')

# Get the height and width of the original image
image_height, image_width = input_image.shape[:2]

# Resize the image for uniform processing
if image_height > 500 and image_width > 500:
    resized_image = cv2.resize(input_image, (700, 700), interpolation=cv2.INTER_AREA)
else:
    resized_image = cv2.resize(input_image, (700, 700), interpolation=cv2.INTER_CUBIC)

# Initialize lists for storing detected shape properties
bounding_boxes = []  # To store bounding box coordinates for each shape
shape_centers = []   # To store the center coordinates of each shape
shape_areas = []     # To store the areas of contours

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Optionally calculate the average brightness of the grayscale image (commented for now)
"""
average_brightness = np.sum(grayscale_image, dtype=np.int32) // (grayscale_image.shape[0] * grayscale_image.shape[1])
print("Average Brightness:", average_brightness)
"""

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(grayscale_image, 125, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on the blank overlay image for visualization
cv2.drawContours(contour_overlay, contours, -1, (0, 0, 255), 1)

# ------------------- Shape Detection and Classification ------------------- #

# Counter for skipping irrelevant contours (e.g., the entire image boundary)
contour_index = 0

for contour in contours:
    # Initialize variables for storing bounding box properties
    sum_x, sum_y = 0, 0
    min_x, min_y = 1000, 1000
    max_x, max_y = 0, 0

    # Calculate the area of the contour
    contour_area = cv2.contourArea(contour)

    # Filter out small or irrelevant contours based on area
    if contour_area > 5000:
        # Ignore the first contour if it's the image border
        if contour_index == 0:
            contour_index += 1
            continue

        # Approximate the contour to a polygon
        approximated_polygon = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True
        )

        # Calculate the centroid (center point) of the shape
        moments = cv2.moments(contour)
        if moments['m00'] != 0.0:
            centroid_x = int(moments['m10'] / moments['m00'])
            centroid_y = int(moments['m01'] / moments['m00'])
            shape_centers.append([centroid_x, centroid_y])

        # Get the bounding box of the contour
        bounding_x, bounding_y, bounding_width, bounding_height = cv2.boundingRect(contour)
        bounding_boxes.append([
            bounding_x, bounding_y, bounding_x + bounding_width, bounding_y + bounding_height, 0
        ])

        # Add the contour area to the list
        shape_areas.append(contour_area)

        # Determine the shape based on the number of sides of the polygon
        num_sides = len(approximated_polygon)
        if num_sides == 3:
            shape_label = 'Triangle'
            color = (0, 0, 255)  # Red
        elif num_sides == 4:
            shape_label = 'Quadrilateral'
            color = (0, 255, 255)  # Yellow
        elif num_sides == 5:
            shape_label = 'Pentagon'
            color = (0, 255, 0)  # Green
        elif num_sides == 6:
            shape_label = 'Hexagon'
            color = (255, 0, 0)  # Blue
        elif num_sides == 8:
            shape_label = 'Octagon'
            color = (255, 0, 255)  # Purple
        else:
            shape_label = 'Circle'
            color = (255, 255, 0)  # Cyan

        # Draw the contour and annotate the shape name
        cv2.drawContours(resized_image, [contour], 0, color, 1)
        cv2.putText(resized_image, shape_label, (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ------------------- Debugging and Outputs ------------------- #

# Print summaries of detected shapes for verification
print("Shape Detection Summary:")
print(f"Bounding Boxes: {bounding_boxes}")
print(f"Shape Centers: {shape_centers}")
print(f"Shape Areas: {shape_areas}")

# Display the processed image and contours
cv2.imshow("Detected Shapes", resized_image)
cv2.imshow("Contours", contour_overlay)

"""
Arrow Detection and Angle Calculation
Description:
    This section identifies arrows in an image based on contour properties such as area and shape.
    It calculates the direction of the arrow by analyzing the centroid and boundary points,
    and determines the angle of inclination using slope calculations and trigonometric functions.
"""

# Arrow identification based on area thresholds
if 1000 < contour_area < 5000:
    # Approximate the contour to a polygon
    approximated_polygon = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True
    )

    # Check if the approximated polygon has a sufficient number of sides to be considered an arrow
    if 6 <= len(approximated_polygon) <= 10:
        # Initialize variables to calculate the bounding box and centroid
        sum_x, sum_y = 0, 0
        min_x, min_y = 1000, 1000
        max_x, max_y = 0, 0

        # Process each vertex of the approximated polygon
        for vertex_index in range(len(approximated_polygon)):
            # Extract x and y coordinates of the vertex
            vertex_x = approximated_polygon[vertex_index][0][0]
            vertex_y = approximated_polygon[vertex_index][0][1]

            # Update sum of coordinates for centroid calculation
            sum_x += vertex_x
            sum_y += vertex_y

            # Update the bounding box coordinates
            min_x = min(min_x, vertex_x)
            min_y = min(min_y, vertex_y)
            max_x = max(max_x, vertex_x)
            max_y = max(max_y, vertex_y)

        # Calculate the centroid and midpoints of the arrow
        side_count = len(approximated_polygon) - 1
        centroid = (sum_x // (side_count + 1), sum_y // (side_count + 1))
        midpoint = ((min_x + max_x) // 2, (min_y + max_y) // 2)
        offset_point = (((max_x + min_x) // 2) + 40, (min_y + max_y) // 2)

        # Draw visual aids on the image for debugging and validation
        cv2.circle(resized_image, centroid, 3, (0, 255, 0), thickness=-1)  # Centroid
        cv2.circle(resized_image, midpoint, 3, (0, 255, 0), thickness=-1)  # Midpoint
        cv2.circle(resized_image, offset_point, 3, (0, 255, 0), thickness=-1)  # Offset point
        cv2.rectangle(resized_image, (min_x - 15, min_y - 15), (max_x + 15, max_y + 15), (10, 40, 80), thickness=1)
        cv2.line(resized_image, midpoint, centroid, (255, 120, 30))  # Line to centroid
        cv2.line(resized_image, midpoint, offset_point, (255, 120, 30))  # Line to offset point

        # Calculate slopes and the angle between the centroid and bounding points
        slope_mid_to_offset = calculate_slope(offset_point, midpoint)
        slope_mid_to_centroid = calculate_slope(centroid, midpoint)
        tangent_value = (slope_mid_to_offset - slope_mid_to_centroid) / (
            1 + slope_mid_to_offset * slope_mid_to_centroid
        )
        angle_in_radians = math.atan(tangent_value)
        angle_in_degrees = math.degrees(angle_in_radians)

        # Determine the angle based on the quadrant of the arrow
        if midpoint[0] >= centroid[0]:
            if midpoint[1] >= centroid[1]:
                angle = 180 + angle_in_degrees
                label_position = (min_x - 15, min_y - 20)
            else:
                angle = 180 + angle_in_degrees
                label_position = (min_x - 15, min_y - 20)
        else:
            if midpoint[1] >= centroid[1]:
                angle = angle_in_degrees
                label_position = (min_x - 15, min_y - 20)
            else:
                angle = 270 - angle_in_degrees
                label_position = (min_x - 15, min_y - 20)

        # Annotate the image with the calculated angle
        cv2.putText(resized_image, f"Angle={angle:.2f}", label_position, cv2.FONT_HERSHEY_COMPLEX, 0.4, (120, 120, 255))

        # Draw additional visualization lines to highlight the arrow's direction
        cv2.line(resized_image, midpoint, (midpoint[0], centroid[1]), (255, 0, 255), thickness=1)
        cv2.line(resized_image, midpoint, (centroid[0], midpoint[1]), (255, 0, 255), thickness=1)
        cv2.line(resized_image, (centroid[0], midpoint[1]), (centroid[0] + 10, midpoint[1] + 10), (255, 0, 255), thickness=1)
        cv2.line(resized_image, (centroid[0], midpoint[1]), (centroid[0] - 10, midpoint[1] - 10), (255, 0, 255), thickness=1)
        cv2.line(resized_image, (midpoint[0], centroid[1]), (midpoint[0] - 10, centroid[1] + 10), (255, 0, 255), thickness=1)
        cv2.line(resized_image, (midpoint[0], centroid[1]), (midpoint[0] + 10, centroid[1] - 10), (255, 0, 255), thickness=1)

        # Append the detected arrow's details to the list for further processing
        arrow_details = [
            centroid[0],  # Centroid x-coordinate
            centroid[1],  # Centroid y-coordinate
            min_x,        # Bounding box minimum x-coordinate
            min_y,        # Bounding box minimum y-coordinate
            max_x,        # Bounding box maximum x-coordinate
            max_y,        # Bounding box maximum y-coordinate
            round(angle, 2)  # Calculated angle (rounded to two decimals)
        ]
        sum_of_arrows.append(arrow_details)

        # Mark the detected arrow on the image
        cv2.circle(resized_image, (min_x, min_y), 2, (10, 10, 255), thickness=-1)  # Mark minimum point
        cv2.putText(resized_image, "Arrow", (min_x, min_y - 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255))

# Display the updated image with annotated arrows and shapes
cv2.imshow('Shapes and Arrows', resized_image)


"""
Elimination of Extra Centers and Noise Reduction
Description:
    This section removes redundant centers and noise in the detected shapes and text data.
    It addresses edge errors, overlaps, and eliminates inaccuracies caused by text recognition.
"""

# Elimination of overlapping and redundant centers
for i in range(len(center_points)):
    overlap_count = 0
    for x in range(len(center_points) - i):
        # Check for overlapping centers within a threshold
        if (-10 < center_points[i][0] - center_points[x + i - overlap_count][0] < 10) and \
           (-10 < center_points[i][1] - center_points[x + i - overlap_count][1] < 10) and \
           (x + i - overlap_count != i):
            center_points.remove(center_points[x + i - overlap_count])
            bounding_boxes.remove(bounding_boxes[x + i - overlap_count])
            detected_areas.remove(detected_areas[x + i - overlap_count])
            overlap_count += 1

    edge_count = 0
    for x in range(len(center_points)):
        count_centers = 0
        for center in center_points:
            # Check if the center is within the boundaries of the bounding box
            if (bounding_boxes[x - edge_count][0] < center[0] < bounding_boxes[x - edge_count][0] + bounding_boxes[x - edge_count][2]) and \
               (bounding_boxes[x - edge_count][1] < center[1] < bounding_boxes[x - edge_count][1] + bounding_boxes[x - edge_count][3]):
                count_centers += 1
        if count_centers >= 3:  # Remove boxes with excessive centers
            center_points.remove(center_points[x - edge_count])
            bounding_boxes.remove(bounding_boxes[x - edge_count])
            detected_areas.remove(detected_areas[x - edge_count])
            edge_count += 1

# Noise reduction in text recognition results
noise_count = 0
for i in range(len(text_data)):
    # Remove invalid or empty text results
    if text_data[i - noise_count][4] in ('\x0c', '\n\x0c', ''):
        text_data.remove(text_data[i - noise_count])
        noise_count += 1

# Combine centers and bounding box edges
center_points_array = np.array(center_points).reshape(len(center_points), 2)  # Reshape center points array
bounding_boxes_array = np.array(bounding_boxes).reshape(len(bounding_boxes), 5)  # Reshape bounding box array

# Concatenate center points and bounding box details
shape_details = np.concatenate((center_points_array, bounding_boxes_array), axis=1).tolist()

# Drawing mg (weight) and N (normal force) for shapes
for shape in shape_details:
    center_x, center_y, _, _, _, bottom_y, _ = shape
    height_center = bottom_y - center_y  # Calculate center height for arrow scaling

    # Draw mg (weight) arrow
    resized_image = cv2.line(resized_image, (center_x, center_y), (center_x, center_y + int(height_center / 1.5)), (255, 0, 0), 3)
    resized_image = cv2.line(resized_image, (center_x, center_y + int(height_center / 1.5)), (center_x - int(center_x / 10), center_y + int(height_center / 1.8)), (255, 0, 0), 3)
    resized_image = cv2.line(resized_image, (center_x, center_y + int(height_center / 1.5)), (center_x + int(center_x / 10), center_y + int(height_center / 1.8)), (255, 0, 0), 3)
    resized_image = cv2.putText(resized_image, "mg", (center_x - 20, center_y + int(height_center / 1.5) + 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)

    # Draw N (normal force) arrow
    resized_image = cv2.line(resized_image, (center_x, center_y), (center_x, center_y - int(height_center / 1.5)), (0, 0, 255), 3)
    resized_image = cv2.line(resized_image, (center_x, center_y - int(height_center / 1.5)), (center_x + int(center_x / 10), center_y - int(height_center / 1.8)), (0, 0, 255), 3)
    resized_image = cv2.line(resized_image, (center_x, center_y - int(height_center / 1.5)), (center_x - int(center_x / 10), center_y - int(height_center / 1.8)), (0, 0, 255), 3)
    resized_image = cv2.putText(resized_image, "N", (center_x - 20, center_y - int(height_center / 1.5) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)


"""
Variable Extraction and Unification
Description:
    This section identifies variables (e.g., weights in "kg", forces in "N", or other variables)
    from the detected text data. It maps these variables to their respective shapes or arrows
    by calculating distances. Finally, it structures the data for Free Body Diagram (FBD) representation.
"""

# Initialize lists to store extracted variable data
weights_in_kg = []  # List to store weights in kilograms
forces_in_N = []    # List to store forces in Newtons
unidentified_indices = []  # Tracks the indices of unidentified variables
variable_index = 0  # Index tracker for variables
unidentified_variables = []  # List to store unidentified variables

# Parse detected text data to extract variable information
for text_entry in text_data:
    """
    Each text_entry contains the bounding box coordinates and detected text:
    - text_entry[0]: Top-left x-coordinate
    - text_entry[1]: Top-left y-coordinate
    - text_entry[2]: Width of the bounding box
    - text_entry[3]: Height of the bounding box
    - text_entry[4]: Detected text
    """

    # Extract the text string and convert to uppercase for uniformity
    text_string = text_entry[4].upper()

    # Find the locations of "KG" and "N" in the string
    kg_location = text_string.find("KG")  # Location of "KG" for weight variables
    n_location = text_string.find("N")   # Location of "N" for force variables

    # Extract numeric values preceding "KG" and "N" in the string
    numeric_kg = text_string[:kg_location]
    numeric_n = text_string[:n_location]

    # Define a regex pattern to isolate numeric values
    number_pattern = re.compile(r'\D')  # Matches non-numeric characters

    # Extract numeric parts for "KG" and "N"
    extracted_kg = number_pattern.split(numeric_kg)
    extracted_n = number_pattern.split(numeric_n)

    # Process "KG" values and store valid entries in weights_in_kg
    kg_value = []
    for value in extracted_kg:
        if value != '' and kg_location != -1:  # Ensure valid numeric value
            kg_value = value
    if kg_value:
        weights_in_kg.append([
            int((text_entry[0] + text_entry[2] / 2) * 1.4),  # Center x-coordinate
            int((text_entry[1] + text_entry[3] / 2) * 1.4),  # Center y-coordinate
            int(kg_value),  # Numeric value
            "kg"  # Unit type
        ])

    # Process "N" values and store valid entries in forces_in_N
    n_value = []
    for value in extracted_n:
        if value != '' and n_location != -1:  # Ensure valid numeric value
            n_value = value
    if n_value:
        forces_in_N.append([
            int((text_entry[0] + text_entry[2] / 2) * 1.4),  # Center x-coordinate
            int((text_entry[1] + text_entry[3] / 2) * 1.4),  # Center y-coordinate
            int(n_value),  # Numeric value
            "N"  # Unit type
        ])

    # Identify variables that are neither "KG" nor "N"
    if not kg_value and not n_value:
        unidentified_indices.append(variable_index)  # Record the index
    variable_index += 1

# Store unidentified variables with their positions
for idx in unidentified_indices:
    unidentified_variables.append([
        int((text_data[idx][0] + text_data[idx][2] / 2) * 1.4),  # Center x-coordinate
        int((text_data[idx][1] + text_data[idx][3] / 2) * 1.4),  # Center y-coordinate
        text_data[idx][4],  # Detected text
        "vari"  # Mark as unidentified variable
    ])

# Combine all identified variables into a single list
print("Unidentified Variables:", unidentified_variables)
all_variables = weights_in_kg + forces_in_N + unidentified_variables
print("All Identified Variables:", all_variables)

# Combine shape data and sum of arrows for unification
combined_data = shape_details + sum_of_arrows  # Merge shape details and arrow data

# Map variables to their closest corresponding shapes or arrows
for shape_entry in combined_data:
    """
    For each shape_entry, find the closest variable from all_variables
    based on Euclidean distance, and append the variable's data to the shape_entry.
    """
    min_distance = float('inf')  # Initialize minimum distance
    closest_index = -1  # Track the index of the closest variable

    for idx, variable_entry in enumerate(all_variables):
        # Calculate Euclidean distance between shape and variable
        x_diff = shape_entry[0] - variable_entry[0]
        y_diff = shape_entry[1] - variable_entry[1]
        distance = math.sqrt(x_diff ** 2 + y_diff ** 2)

        # Update the closest variable if the distance is smaller
        if distance < min_distance:
            min_distance = distance
            closest_index = idx

    # Append the closest variable's value and type to the shape_entry
    shape_entry.insert(-1, all_variables[closest_index][2])  # Variable value
    shape_entry.insert(-1, all_variables[closest_index][3])  # Variable type

# Define a class for Free Body Diagram (FBD) objects
class FreeBodyDiagram:
    """
    A class representing a Free Body Diagram (FBD) object.
    Attributes:
        mass: The mass associated with the FBD object.
    """
    def __init__(self, mass):
        self.mass = mass

# Finalize the data structure for FBD representation
final_fbd_data = []
for entry in combined_data:
    """
    Separate entries based on the variable type:
    - If the variable type is "kg", treat it as a weight.
    - Otherwise, include it in the final FBD data.
    """
    if entry[-2] == "kg":
        final_fbd_data.append(entry)
    else:
        final_fbd_data.append(entry)

# Display results for debugging and verification
print("Combined Data:", combined_data)
print("Final FBD Data:", final_fbd_data)



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