# import cv2
# import numpy as np
# import mediapipe as mp
# from collections import deque

# # Function to simulate shape prediction (Placeholder)
# def predict_shape(contour):
#     # This function should contain your model's prediction logic
#     # For this example, we will return a simple label based on the number of corners
#     num_corners = len(contour)
#     if num_corners == 3:
#         return "Triangle"
#     elif num_corners == 4:
#         return "Square/Rectangle"
#     elif num_corners > 4:
#         return "Circle"
#     return "Unknown"

# # Initialize webcam and drawing settings
# cap = cv2.VideoCapture(0)
# paintWindow = np.zeros((480, 640, 3)) + 255  # White canvas
# drawing = False
# points = deque(maxlen=512)
# color = (0, 0, 255)  # Red color for drawing
# thickness = 5

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)  # Flip frame horizontally for mirror effect
#     cv2.rectangle(frame, (0, 0), (640, 480), (255, 255, 255), -1)  # White background

#     # Drawing logic
#     if drawing:
#         cv2.line(paintWindow, points[-1], (x, y), color, thickness)

#     cv2.imshow("Drawing", paintWindow)
#     cv2.imshow("Webcam", frame)

#     # Shape detection
#     gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#         cv2.drawContours(frame, [approx], 0, (0), 5)

#         # Call your shape prediction function here
#         detected_shape = predict_shape(approx)
        
#         x = approx.ravel()[0]
#         y = approx.ravel()[1] - 5

#         # Display detected shape on the frame
#         cv2.putText(frame, detected_shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('d'):  # Press 'd' to start drawing
#         drawing = True
#     elif key == ord('r'):  # Press 'r' to reset
#         paintWindow = np.zeros((480, 640, 3)) + 255
#         points.clear()
#     elif key == ord('q'):  # Press 'q' to quit
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import pytesseract
from collections import deque

# Function for shape detection
def predict_shape(contour):
    num_corners = len(contour)
    if num_corners == 3:
        return "Triangle"
    elif num_corners == 4:
        return "Square/Rectangle"
    elif num_corners > 4:
        return "Circle"
    return "Unknown"

# Function for alphanumeric character recognition using Tesseract OCR
def predict_alphanumeric(contour, paintWindow):
    mask = np.zeros_like(paintWindow)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(paintWindow, mask)
    
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    custom_config = r'--oem 3 --psm 8'
    alphanumeric = pytesseract.image_to_string(thresh, config=custom_config)
    return alphanumeric.strip()

# Initialize webcam, paint window, and settings
cap = cv2.VideoCapture(0)
paintWindow = np.zeros((480, 640, 3)) + 255  # White canvas
drawing = False
points = deque(maxlen=512)
color = (0, 0, 255)
thickness = 5
mode = 'shape'  # Start with shape detection mode

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (0, 0), (640, 480), (255, 255, 255), -1)

    # Drawing logic
    if drawing:
        cv2.line(paintWindow, points[-1], (x, y), color, thickness)

    cv2.imshow("Drawing", paintWindow)
    cv2.imshow("Webcam", frame)

    # Detection logic
    gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], 0, (0), 5)

        # Switch between shape and alphanumeric modes
        if mode == 'shape':
            detected_shape = predict_shape(approx)
            display_text = detected_shape
        elif mode == 'alphanumeric':
            detected_char = predict_alphanumeric(approx, paintWindow)
            display_text = detected_char
        
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        cv2.putText(frame, display_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Key bindings for mode switching and drawing
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):  # Press 'd' to start drawing
        drawing = True
    elif key == ord('r'):  # Press 'r' to reset the canvas
        paintWindow = np.zeros((480, 640, 3)) + 255
        points.clear()
    elif key == ord('a'):  # Switch to alphanumeric detection
        mode = 'alphanumeric'
        print("Switched to Alphanumeric Detection Mode")
    elif key == ord('s'):  # Switch to shape detection
        mode = 'shape'
        print("Switched to Shape Detection Mode")
    elif key == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
