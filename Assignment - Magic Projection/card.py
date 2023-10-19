import cv2
import numpy as np

# Define a function to calculate the equation of a line
def line_equation(x1, y1, x2, y2):
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b
    else:
        return None

# Define a function to check if a point (x, y) is on the left side of the line
def is_on_left_side(x, y, line_m, line_b):
    if line_m is not None:
        expected_x = (x - line_b) / line_m
        return y < expected_x
    return False

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify a video file path

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    x1_sum = 0
    y1_sum = 0
    x2_sum = 0
    y2_sum = 0
    lines_count = 0

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta)

            # Filter lines within the angle
            if  angle >= 120 or angle <= 60:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1500 * (-b))
                y1 = int(y0 + 1500 * (a))
                x2 = int(x0 - 1500 * (-b))
                y2 = int(y0 - 1500 * (a))

                # Draw detected lines on the original frame
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Sum x, y position
                x1_sum += x1
                y1_sum += y1
                x2_sum += x2
                y2_sum += y2
                lines_count += 1

    if lines_count > 0:
        # Average x, y position
        x1_avg = int(x1_sum / lines_count)
        y1_avg = int(y1_sum / lines_count)
        x2_avg = int(x2_sum / lines_count)
        y2_avg = int(y2_sum / lines_count)

        # Draw average detected lines on the original frame
        cv2.line(frame, (x1_avg, y1_avg), (x2_avg, y2_avg), (0, 255, 0), 2)

        line_m, line_b = line_equation(x1_avg, y1_avg, x2_avg, y2_avg)
        print(line_m, line_b)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for the red color in BGR format
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask to identify red pixels
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Get the non-zero positions (coordinates) in the mask
    non_zero_positions = np.argwhere(mask > 0)
    
    # Iterate through non-zero positions
    for x, y in non_zero_positions:
        print(is_on_left_side(x, y, line_m, line_b))
        if is_on_left_side(x, y, line_m, line_b):
            frame[x, y] = [0, 0, 255]  # Replace red with blue for points on the left side

    # Display the resulting frame
    cv2.imshow('Line Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
