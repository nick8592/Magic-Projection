import os
import sys
import cv2
import math
import numpy as np
import itertools

def generate_combinations(target_sum, num_elements):
    number_range = list(range(1, 7))
    combinations = []

    for combo in itertools.combinations_with_replacement(number_range, num_elements):
        if sum(combo) == target_sum:
            combinations.append(combo)

    return combinations

# Generate combinations for a target_sum using numbers of elements
if len(sys.argv) != 2:
    print("Usage: python dice.py <target_sum>")
    sys.exit(1)

try:
    target_sum = int(sys.argv[1])
except ValueError:
    print("target_sum must be integers")
    sys.exit(1)

# Load custom dice images
dice_images = {}
for i in range(1, 7):
    img_path = os.path.join("dice-images", f"{i}.png")
    dice_images[i] = cv2.imread(img_path)

# Load the image
image = cv2.imread("dice-dataset/0a0e52b1-115f-42ea-be8c-cc78b52f2b21.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Use Canny edge detection to find edges in the image
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store rotated bounding boxes
dice_boxes = []

# Set a threshold for minimum dice size
min_dice_size = 45

for contour in contours:
    # Calculate the area of each contour
    area = cv2.contourArea(contour)

    if area > min_dice_size:
        # Approximate the contour to a polygon (to handle rotated bounding boxes)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) >= 4:
            # If the polygon has 4 corners, it's likely a dice
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            dice_boxes.append(box)

# Draw rotated bounding boxes around the detected dice
for box in dice_boxes:
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# Draw red dots at the vertices of the detected polygons
for box in dice_boxes:
    for point in box:
        cv2.circle(image, tuple(point), 2, (0, 0, 255), -1)

# Display the combinations
num_elements = len(dice_boxes)
combinations = generate_combinations(target_sum, num_elements)
for combo in combinations:
    print(combo)

# TODO: Project new number on the dices
# # Replace detected dice with custom dice images
# for i, box in enumerate(dice_boxes):
#     print(i)
#     p1, p2, p3, p4 = box
#     dice_number = combinations[0][i]  # The detected dice number (you may need to adjust this)

#     if dice_number in dice_images:
#         custom_dice = dice_images[dice_number]

#         # Calculate the length of the square's side
#         side_length =int(math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2))

#         width = abs(p1[0] - p3[0])
#         height = abs(p1[1] - p3[1])
#         print(width)
#         print(height)

#         # Resize the custom dice image to match the dimensions of the detected dice box
#         custom_dice = cv2.resize(custom_dice, (width, height))
        
#         # Replace the detected dice with the custom dice image
#         image[int(p1[1]):int(p3[1]), int(p1[0]):int(p3[0])] = custom_dice

# Display the image with rotated bounding boxes
cv2.imshow("Dice Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
