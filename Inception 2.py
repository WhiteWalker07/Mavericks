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
import re
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


"""
Mass and Angle Storage with Force Components
Description:
    This section processes detected data to:
    1. Associate forces and angles with corresponding bodies.
    2. Compute the components of forces (X and Y directions) applied on each body.
    3. Identify unknown variables for further processing.
"""

# Initialize a list to track entries to be removed
entries_to_remove = []

# Loop through the final list of objects and forces to associate mass or unknown variables
for index, entry in enumerate(final_fbd_data):
    # Check if the entry is a force or an unknown variable
    if entry[7] == "N" or entry[7] == "vari":
        entries_to_remove.append(index)  # Mark the entry for removal
        is_first = True  # Flag for the first distance comparison
        closest_index = 0  # Index of the closest mass entry
        min_distance = float('inf')  # Initialize minimum distance

        # Iterate through the final data to find the closest mass
        for mass_index, mass_entry in enumerate(final_fbd_data):
            if mass_entry[7] == "kg":  # Only compare with mass entries
                # Calculate the distances from the force to the front and back of the mass
                front_distance = math.sqrt((mass_entry[0] - entry[2])**2 + (mass_entry[1] - entry[3])**2)
                back_distance = math.sqrt((mass_entry[0] - entry[4])**2 + (mass_entry[1] - entry[5])**2)

                # Select the smaller distance
                distance = min(front_distance, back_distance)

                # Update the closest mass if a smaller distance is found
                if not is_first and distance < min_distance:
                    min_distance = distance
                    closest_index = mass_index
                elif is_first:
                    min_distance = distance
                    closest_index = 0
                    is_first = False

        # Append the force data to the closest mass entry
        final_fbd_data[closest_index].extend([entry[-3], entry[-2], entry[-1]])

# Remove entries that have been associated with masses
for i, index in enumerate(entries_to_remove):
    final_fbd_data.pop(index - i)

print("The final compressed form with mass and forces associated:\n", final_fbd_data)

# --- Compute Force Components for Each Body ---
force_components = []  # List to store resultant force components for each body
unknown_variables = []  # List to track unknown variables for resolution

# Process each body in the final data
for body_index, body_entry in enumerate(final_fbd_data):
    total_entries = len(body_entry)
    if total_entries >= 10:  # Ensure sufficient data is present for force computation
        resultant_x = 0  # X-component of the resultant force
        resultant_y = 0  # Y-component of the resultant force

        # Loop through the forces associated with the body
        for i in range((total_entries - 9) // 3):
            force_magnitude = body_entry[9 + i * 3]
            force_angle = math.radians(body_entry[11 + i * 3])  # Convert angle to radians

            if body_entry[10 + i * 3] == "N":  # Process forces
                # Compute force components
                resultant_x += force_magnitude * math.cos(force_angle)
                resultant_y += force_magnitude * math.sin(force_angle)
            elif body_entry[10 + i * 3] == "vari":  # Track unknown variables
                unknown_variables.append([body_index, force_magnitude, body_entry[11 + i * 3]])

        # Store the resultant force components
        force_components.append([resultant_x, resultant_y])

print("Resultant force components for each body (X, Y):", force_components)
print("Unknown variables requiring resolution:", unknown_variables)

# --- Function to Extract Data for Unknown Variables ---
def extract_variable_data():
    """
    Extract additional information about unknown variables from the user.
    Prompts the user to resolve angles or provide missing data for variables.
    """
    # Predefined set of standard angles for matching
    standard_angles = [0, 30, 45, 60, 90, 120, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360]

    # Iterate through unknown variables
    for variable in unknown_variables:
        angle_differences = [abs(variable[2] - angle) for angle in standard_angles]
        closest_angle = standard_angles[angle_differences.index(min(angle_differences))]
        print(f"Closest standard angle for variable {variable[1]}: {closest_angle}")

        # Prompt user for additional details
        user_input = int(input(
            f"Information related to variable {variable[1]}:\n"
            "1. Angle is given.\n"
            "2. No information is available.\n"
            "3. Both value and angle are known.\n"
            "4. The variable is irrelevant.\n"
            "Enter the most suitable option: "
        ))

        # Process user input
        if user_input == 3:
            value = input(f"Enter the value for variable {variable[1]} (with unit): ")
            angle = input(f"Enter the angle for variable {variable[1]}: ")
        elif user_input == 1:
            angle = input(f"Enter the angle for variable {variable[1]}: ")
        elif user_input == 2:
            print("No information provided for the variable.")
        elif user_input == 4:
            print(f"Variable {variable[1]} marked as irrelevant.")
        else:
            print("Invalid option. Please try again.")
            extract_variable_data()

# Print outputs for verification
print("Force components (final):", force_components)
print("Unknown variables:", unknown_variables)


# SURFACE IDENTIFICATION
# This section calculates the normal forces and friction forces acting on each object.
# An object is considered "above" another if its center lies between the minimum and maximum x-coordinates of the lower object
# and within the vertical bounds of the lower object's rectangle.

normal_forces = []  # To store calculated normal forces for each object
friction_forces = []  # To store calculated friction forces for each object
object_index = 0  # Index to track the current object

# Iterate over all objects (referred to as "upper objects")
for upper_object in final_fbd_data:
    # Calculate initial normal force (weight of the object minus Y-component of forces)
    normal_force = upper_object[6] * 10 - force_components[object_index][1]
    total_friction_force = force_components[object_index][0]  # Start with X-component of forces

    # Compare with all other objects (referred to as "lower objects")
    for lower_index, lower_object in enumerate(final_fbd_data):
        if (upper_object[2] < lower_object[0] < upper_object[4]) and \
           (0 < lower_object[1] < upper_object[3]) and (object_index != lower_index):
            # Add contribution of the lower object's normal force
            normal_force += lower_object[6] * 10 - force_components[lower_index][1]
            # Add friction force contribution from the lower object
            total_friction_force += force_components[lower_index][0]

    object_index += 1
    # Append the calculated forces
    normal_forces.append(float(f"{normal_force:.5f}"))
    friction_forces.append(float(f"{total_friction_force:.5f}") if total_friction_force > 0 else float(f"{total_friction_force:.6f}"))

# Display calculated normal and friction forces
print("Normal Forces: ", normal_forces)
print("Friction Forces Required to Keep the System Static: ", friction_forces)

# Display values of normal forces and weights on the image
counter = 0
for shape_entry in object_shapes:
    x_center = shape_entry[0]
    y_center_top = shape_entry[1] - int((shape_entry[5] - shape_entry[3]) // 2) + 20
    y_center_bottom = shape_entry[1] + int((shape_entry[5] - shape_entry[3]) // 2) - 10

    # Display normal force and weight
    image = cv2.putText(image, f"   ={shape_entry[6] * 10} N", (x_center - 20, y_center_bottom), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
    image = cv2.putText(image, f"  ={normal_forces[counter]} N", (x_center - 20, y_center_top), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
    counter += 1

# Calculate the coefficient of friction for each object
coefficients_of_friction = []  # To store calculated coefficients of friction
for index, normal_force in enumerate(normal_forces):
    friction = friction_forces[index]
    coefficient = abs(friction / normal_force)
    coefficients_of_friction.append(float(f"{coefficient:.5f}"))

print("Coefficient of Friction for the respective bodies: ", coefficients_of_friction)

# Visualize friction and coefficients of friction on the image
for index, friction_force in enumerate(friction_forces):
    if friction_force > 0:
        # Visualize friction force (positive)
        image = cv2.line(image, (final_fbd_data[index][2], final_fbd_data[index][5]), 
                         (final_fbd_data[index][2] - 60, final_fbd_data[index][5]), (120, 0, 180), thickness=2)
        image = cv2.line(image, (final_fbd_data[index][2] - 60, final_fbd_data[index][5]), 
                         (final_fbd_data[index][2] - 50, final_fbd_data[index][5] + 10), (120, 0, 180), thickness=2)
        image = cv2.line(image, (final_fbd_data[index][2] - 60, final_fbd_data[index][5]), 
                         (final_fbd_data[index][2] - 50, final_fbd_data[index][5] - 10), (120, 0, 180), thickness=2)
        image = cv2.putText(image, f"{friction_force} N", (final_fbd_data[index][2] - 130, final_fbd_data[index][5]), 
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        image = cv2.putText(image, f"u={coefficients_of_friction[index]}", 
                            (final_fbd_data[index][2] - 100, final_fbd_data[index][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    elif friction_force < 0:
        # Visualize friction force (negative)
        image = cv2.line(image, (final_fbd_data[index][4], final_fbd_data[index][5]), 
                         (final_fbd_data[index][4] + 100, final_fbd_data[index][5]), (120, 0, 180), thickness=2)
        image = cv2.line(image, (final_fbd_data[index][4] + 100, final_fbd_data[index][5]), 
                         (final_fbd_data[index][4] + 90, final_fbd_data[index][5] + 10), (120, 0, 180), thickness=2)
        image = cv2.line(image, (final_fbd_data[index][4] + 100, final_fbd_data[index][5]), 
                         (final_fbd_data[index][4] + 90, final_fbd_data[index][5] - 10), (120, 0, 180), thickness=2)
        image = cv2.putText(image, f"{abs(friction_force)} N", 
                            (final_fbd_data[index][4] + 110, final_fbd_data[index][5]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        image = cv2.putText(image, f"u={coefficients_of_friction[index]}", 
                            (final_fbd_data[index][4] + 150, final_fbd_data[index][1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

# Save the image and analysis results if requested by the user
save_choice = input("Do you want to save the FBD analysis to your device? (Y/N): ").strip().upper()

# Save results if the user agrees
if save_choice == "Y":
    folder_name = input("Enter the folder name to save the files: ")
    save_directory = r"save dir"  # Function to prompt user for save location

    os.makedirs(os.path.join(save_directory, folder_name), exist_ok=True)
    save_path = os.path.join(save_directory, folder_name)

    # Save the annotated image
    cv2.imwrite(os.path.join(save_path, "OUTPUT_IMG.png"), image)

    # Save the analysis results in a text file
    with open(os.path.join(save_path, "OUTPUT_TXT.txt"), "w") as result_file:
        for idx, components in enumerate(force_components):
            result_file.write(f"Analysis of Object {idx + 1}:\n")
            result_file.write(f"Equivalent X-Component: {components[0]:.5f}\n")
            result_file.write(f"Equivalent Y-Component: {components[1]:.5f}\n")
            result_file.write(f"Normal Reaction: {normal_forces[idx]}\n")
            result_file.write(f"Minimum Coefficient of Friction: {coefficients_of_friction[idx]}\n\n")

# Cleanup and finalize
os.remove("recognized.txt")
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()