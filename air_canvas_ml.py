# # # # All the imports go here
# # # import cv2
# # # import numpy as np
# # # import mediapipe as mp
# # # from collections import deque
# # # import pytesseract  # Import pytesseract for OCR

# # # # Set the tesseract cmd path if needed (e.g., for Windows)
# # # # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # # # Giving different arrays to handle colour points of different colour
# # # bpoints = [deque(maxlen=1024)]
# # # gpoints = [deque(maxlen=1024)]
# # # rpoints = [deque(maxlen=1024)]
# # # ypoints = [deque(maxlen=1024)]

# # # # These indexes will be used to mark the points in particular arrays of specific colour
# # # blue_index = 0
# # # green_index = 0
# # # red_index = 0
# # # yellow_index = 0

# # # # Kernel for dilation
# # # kernel = np.ones((5, 5), np.uint8)

# # # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
# # # colorIndex = 0

# # # # Canvas setup
# # # paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
# # # paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
# # # paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# # # cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # # cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# # # # Initialize mediapipe
# # # mpHands = mp.solutions.hands
# # # hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# # # mpDraw = mp.solutions.drawing_utils

# # # # Initialize the webcam
# # # cap = cv2.VideoCapture(0)
# # # ret = True
# # # while ret:
# # #     # Read each frame from the webcam
# # #     ret, frame = cap.read()

# # #     x, y, c = frame.shape

# # #     # Flip the frame vertically
# # #     frame = cv2.flip(frame, 1)
# # #     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # #     # Draw UI buttons
# # #     frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
# # #     frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
# # #     frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
# # #     frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
# # #     frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
# # #     cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # #     cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# # #     # Get hand landmark prediction
# # #     result = hands.process(framergb)

# # #     # Post process the result
# # #     if result.multi_hand_landmarks:
# # #         landmarks = []
# # #         for handslms in result.multi_hand_landmarks:
# # #             for lm in handslms.landmark:
# # #                 lmx = int(lm.x * 640)
# # #                 lmy = int(lm.y * 480)
# # #                 landmarks.append([lmx, lmy])

# # #             # Drawing landmarks on frames
# # #             mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

# # #         fore_finger = (landmarks[8][0], landmarks[8][1])
# # #         center = fore_finger
# # #         thumb = (landmarks[4][0], landmarks[4][1])
# # #         cv2.circle(frame, center, 3, (0, 255, 0), -1)

# # #         # Clear the canvas if the thumb is close to the forefinger
# # #         if (thumb[1] - center[1] < 30):
# # #             bpoints.append(deque(maxlen=512))
# # #             blue_index += 1
# # #             gpoints.append(deque(maxlen=512))
# # #             green_index += 1
# # #             rpoints.append(deque(maxlen=512))
# # #             red_index += 1
# # #             ypoints.append(deque(maxlen=512))
# # #             yellow_index += 1

# # #         elif center[1] <= 65:
# # #             if 40 <= center[0] <= 140:  # Clear Button
# # #                 bpoints = [deque(maxlen=512)]
# # #                 gpoints = [deque(maxlen=512)]
# # #                 rpoints = [deque(maxlen=512)]
# # #                 ypoints = [deque(maxlen=512)]
# # #                 blue_index = 0
# # #                 green_index = 0
# # #                 red_index = 0
# # #                 yellow_index = 0
# # #                 paintWindow[67:, :, :] = 255
# # #             elif 160 <= center[0] <= 255:
# # #                 colorIndex = 0  # Blue
# # #             elif 275 <= center[0] <= 370:
# # #                 colorIndex = 1  # Green
# # #             elif 390 <= center[0] <= 485:
# # #                 colorIndex = 2  # Red
# # #             elif 505 <= center[0] <= 600:
# # #                 colorIndex = 3  # Yellow
# # #         else:
# # #             if colorIndex == 0:
# # #                 bpoints[blue_index].appendleft(center)
# # #             elif colorIndex == 1:
# # #                 gpoints[green_index].appendleft(center)
# # #             elif colorIndex == 2:
# # #                 rpoints[red_index].appendleft(center)
# # #             elif colorIndex == 3:
# # #                 ypoints[yellow_index].appendleft(center)

# # #     # Draw lines of all the colors on the canvas and frame
# # #     points = [bpoints, gpoints, rpoints, ypoints]
# # #     for i in range(len(points)):
# # #         for j in range(len(points[i])):
# # #             for k in range(len(points[i][j])):
# # #                 if len(points[i][j]) > 0:
# # #                     if i == 0:
# # #                         cv2.circle(paintWindow, (points[i][j][k][0], points[i][j][k][1]), 3, colors[i], cv2.FILLED)
# # #                     else:
# # #                         cv2.circle(paintWindow, (points[i][j][k][0], points[i][j][k][1]), 3, colors[i], cv2.FILLED)

# # #     # Convert canvas to grayscale for OCR
# # #     gray_canvas = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
# # #     # Thresholding to get a binary image for OCR
# # #     _, binary_canvas = cv2.threshold(gray_canvas, 150, 255, cv2.THRESH_BINARY_INV)

# # #     # Use pytesseract to recognize text
# # #     recognized_text = pytesseract.image_to_string(binary_canvas, config='--psm 10')
# # #     # Draw the recognized text on the original frame
# # #     cv2.putText(frame, f'Recognized: {recognized_text}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# # #     # Display the result
# # #     cv2.imshow("Paint", paintWindow)
# # #     cv2.imshow("Original", frame)

# # #     # Break the loop if 'q' is pressed
# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()




# # # */-+

# # # All the imports go here
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# # from collections import deque

# # # Initialize color points for different colors
# # bpoints = [deque(maxlen=1024)]
# # gpoints = [deque(maxlen=1024)]
# # rpoints = [deque(maxlen=1024)]
# # ypoints = [deque(maxlen=1024)]

# # # Color indices
# # blue_index = 0
# # green_index = 0
# # red_index = 0
# # yellow_index = 0

# # # Kernel for dilation
# # kernel = np.ones((5, 5), np.uint8)

# # # Colors for drawing
# # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
# # colorIndex = 0

# # # Canvas setup
# # paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
# # paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
# # paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
# # paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
# # paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
# # paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# # # Text labels for color selection
# # cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# # cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# # # Initialize mediapipe
# # mpHands = mp.solutions.hands
# # hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# # mpDraw = mp.solutions.drawing_utils

# # # Initialize the webcam
# # cap = cv2.VideoCapture(0)
# # ret = True
# # while ret:
# #     # Read each frame from the webcam
# #     ret, frame = cap.read()

# #     x, y, c = frame.shape

# #     # Flip the frame vertically
# #     frame = cv2.flip(frame, 1)
# #     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     # Draw UI buttons
# #     frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
# #     frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
# #     frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
# #     frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
# #     frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
# #     cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# #     cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

# #     # Get hand landmark prediction
# #     result = hands.process(framergb)

# #     # Post process the result
# #     if result.multi_hand_landmarks:
# #         landmarks = []
# #         for handslms in result.multi_hand_landmarks:
# #             for lm in handslms.landmark:
# #                 lmx = int(lm.x * 640)
# #                 lmy = int(lm.y * 480)
# #                 landmarks.append([lmx, lmy])

# #             # Drawing landmarks on frames
# #             mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

# #         fore_finger = (landmarks[8][0], landmarks[8][1])
# #         center = fore_finger
# #         thumb = (landmarks[4][0], landmarks[4][1])
# #         cv2.circle(frame, center, 3, (0, 255, 0), -1)

# #         # Clear the canvas if the thumb is close to the forefinger
# #         if (thumb[1] - center[1] < 30):
# #             bpoints.append(deque(maxlen=512))
# #             blue_index += 1
# #             gpoints.append(deque(maxlen=512))
# #             green_index += 1
# #             rpoints.append(deque(maxlen=512))
# #             red_index += 1
# #             ypoints.append(deque(maxlen=512))
# #             yellow_index += 1

# #         elif center[1] <= 65:
# #             if 40 <= center[0] <= 140:  # Clear Button
# #                 bpoints = [deque(maxlen=512)]
# #                 gpoints = [deque(maxlen=512)]
# #                 rpoints = [deque(maxlen=512)]
# #                 ypoints = [deque(maxlen=512)]
# #                 blue_index = 0
# #                 green_index = 0
# #                 red_index = 0
# #                 yellow_index = 0
# #                 paintWindow[67:, :, :] = 255
# #             elif 160 <= center[0] <= 255:
# #                 colorIndex = 0  # Blue
# #             elif 275 <= center[0] <= 370:
# #                 colorIndex = 1  # Green
# #             elif 390 <= center[0] <= 485:
# #                 colorIndex = 2  # Red
# #             elif 505 <= center[0] <= 600:
# #                 colorIndex = 3  # Yellow
# #         else:
# #             if colorIndex == 0:
# #                 bpoints[blue_index].appendleft(center)
# #             elif colorIndex == 1:
# #                 gpoints[green_index].appendleft(center)
# #             elif colorIndex == 2:
# #                 rpoints[red_index].appendleft(center)
# #             elif colorIndex == 3:
# #                 ypoints[yellow_index].appendleft(center)

# #     # Draw lines of all the colors on the canvas and frame
# #     points = [bpoints, gpoints, rpoints, ypoints]
# #     for i in range(len(points)):
# #         for j in range(len(points[i])):
# #             for k in range(1, len(points[i][j])):
# #                 if points[i][j][k - 1] is None or points[i][j][k] is None:
# #                     continue
# #                 cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
# #                 cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

# #     # Shape and symbol detection and classification
# #     gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
# #     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
# #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# #     for cnt in contours:
# #         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
# #         # Calculate the area
# #         area = cv2.contourArea(cnt)

# #         if area > 100:  # Filter out small areas
# #             if len(approx) == 3:
# #                 cv2.putText(frame, "Triangle", (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #                 cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
# #             elif len(approx) == 4:
# #                 x, y, w, h = cv2.boundingRect(approx)
# #                 aspectRatio = float(w) / h
# #                 if aspectRatio >= 0.95 and aspectRatio <= 1.05:
# #                     cv2.putText(frame, "Square", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #                 else:
# #                     cv2.putText(frame, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #                 cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
# #             elif len(approx) == 5:
# #                 cv2.putText(frame, "Pentagon", (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #                 cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)
# #             elif len(approx) > 5:
# #                 cv2.putText(frame, "Circle", (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #                 cv2.drawContours(frame, [approx], 0, (0, 0, 255), 2)

# #             # Detect mathematical symbols
# #             symbol_approx = cv2.contourArea(approx)
# #             if area < 1000:  # Assuming small area contours are symbols
# #                 if len(approx) == 4:  # Simple rectangle detection for symbols
# #                     cv2.putText(frame, "Symbol", (approx[0][0][0], approx[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
# #                     cv2.drawContours(frame, [approx], 0, (255, 0, 0), 2)  # Draw symbols in blue

# #     # Show the frames
# #     cv2.imshow("Paint", paintWindow)
# #     cv2.imshow("Frame", frame)

# #     # Break the loop on 'q' key press
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # # Release the camera and destroy all windows
# # cap.release()
# # cv2.destroyAllWindows()



# # 2.pracisecode

# # All the imports go here
# import cv2
# import numpy as np
# import mediapipe as mp
# from collections import deque

# # Giving different arrays to handle color points of different colors
# bpoints = [deque(maxlen=1024)]
# gpoints = [deque(maxlen=1024)]
# rpoints = [deque(maxlen=1024)]
# ypoints = [deque(maxlen=1024)]

# # These indexes will be used to mark the points in particular arrays of specific colors
# blue_index = 0
# green_index = 0
# red_index = 0
# yellow_index = 0

# # Kernel for dilation
# kernel = np.ones((5, 5), np.uint8)

# colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
# colorIndex = 0

# # Canvas setup
# paintWindow = np.zeros((471, 636, 3), dtype=np.uint8) + 255
# paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
# paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), (255, 0, 0), 2)
# paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), (0, 255, 0), 2)
# paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), (0, 0, 255), 2)
# paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), (0, 255, 255), 2)

# cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
# cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# # Initialize mediapipe
# mpHands = mp.solutions.hands
# hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
# mpDraw = mp.solutions.drawing_utils

# # Initialize the webcam
# cap = cv2.VideoCapture(0)
# ret = True
# while ret:
#     # Read each frame from the webcam
#     ret, frame = cap.read()

#     x, y, c = frame.shape

#     # Flip the frame vertically
#     frame = cv2.flip(frame, 1)
#     framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Draw UI buttons
#     frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
#     frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), 2)
#     frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), 2)
#     frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 0, 255), 2)
#     frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 255, 255), 2)
#     cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
#     cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

#     # Get hand landmark prediction
#     result = hands.process(framergb)

#     # Post process the result
#     if result.multi_hand_landmarks:
#         landmarks = []
#         for handslms in result.multi_hand_landmarks:
#             for lm in handslms.landmark:
#                 lmx = int(lm.x * 640)
#                 lmy = int(lm.y * 480)
#                 landmarks.append([lmx, lmy])

#             # Drawing landmarks on frames
#             mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

#         fore_finger = (landmarks[8][0], landmarks[8][1])
#         center = fore_finger
#         thumb = (landmarks[4][0], landmarks[4][1])
#         cv2.circle(frame, center, 3, (0, 255, 0), -1)

#         # Clear the canvas if the thumb is close to the forefinger
#         if (thumb[1] - center[1] < 30):
#             bpoints.append(deque(maxlen=512))
#             blue_index += 1
#             gpoints.append(deque(maxlen=512))
#             green_index += 1
#             rpoints.append(deque(maxlen=512))
#             red_index += 1
#             ypoints.append(deque(maxlen=512))
#             yellow_index += 1

#         elif center[1] <= 65:
#             if 40 <= center[0] <= 140:  # Clear Button
#                 bpoints = [deque(maxlen=512)]
#                 gpoints = [deque(maxlen=512)]
#                 rpoints = [deque(maxlen=512)]
#                 ypoints = [deque(maxlen=512)]
#                 blue_index = 0
#                 green_index = 0
#                 red_index = 0
#                 yellow_index = 0
#                 paintWindow[67:, :, :] = 255
#             elif 160 <= center[0] <= 255:
#                 colorIndex = 0  # Blue
#             elif 275 <= center[0] <= 370:
#                 colorIndex = 1  # Green
#             elif 390 <= center[0] <= 485:
#                 colorIndex = 2  # Red
#             elif 505 <= center[0] <= 600:
#                 colorIndex = 3  # Yellow
#         else:
#             if colorIndex == 0:
#                 bpoints[blue_index].appendleft(center)
#             elif colorIndex == 1:
#                 gpoints[green_index].appendleft(center)
#             elif colorIndex == 2:
#                 rpoints[red_index].appendleft(center)
#             elif colorIndex == 3:
#                 ypoints[yellow_index].appendleft(center)

#     # Draw lines of all the colors on the canvas and frame
#     points = [bpoints, gpoints, rpoints, ypoints]
#     for i in range(len(points)):
#         for j in range(len(points[i])):
#             for k in range(1, len(points[i][j])):
#                 if points[i][j][k - 1] is None or points[i][j][k] is None:
#                     continue
#                 cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
#                 cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

#     # Shape detection and classification
#     gray = cv2.cvtColor(paintWindow, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#         if len(approx) == 3:
#             # Triangle
#             cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 2)
#             cv2.putText(frame, "Triangle", (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#         elif len(approx) == 4:
#             # Rectangle or Square
#             x, y, w, h = cv2.boundingRect(cnt)
#             aspectRatio = float(w) / h
#             if aspectRatio >= 0.95 and aspectRatio <= 1.05:
#                 cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 2)
#                 cv2.putText(frame, "Square", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#             else:
#                 cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 2)
#                 cv2.putText(frame, "Rectangle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#         elif len(approx) == 5:
#             # Pentagon
#             cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 2)
#             cv2.putText(frame, "Pentagon", (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#         elif len(approx) == 6:
#             # Hexagon
#             cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 2)
#             cv2.putText(frame, "Hexagon", (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#         elif len(approx) > 6:
#             # Circle
#             cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 2)
#             cv2.putText(frame, "Circle", (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

#     # Displaying the frame and the paint window
#     cv2.imshow('Paint', paintWindow)
#     cv2.imshow('Frame', frame)

#     # Break the loop when 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()







import cv2
import numpy as np
import pytesseract

# Set the Tesseract executable path (update this path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Parameters for drawing on the air canvas
draw_color = (255, 0, 0)
thickness = 10
background_color = (0, 0, 0)

# Initialize canvas and kernel for morphological operations
canvas = None
kernel = np.ones((5, 5), np.uint8)

# Function to preprocess the image for OCR
def preprocess_for_ocr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh_image

# Start video capture
cap = cv2.VideoCapture(0)

# Set canvas size based on video capture
ret, frame = cap.read()
if ret:
    canvas = np.zeros_like(frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for easier drawing
    frame = cv2.flip(frame, 1)

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color range for detecting marker (e.g., blue color)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.circle(frame, (x + w // 2, y + h // 2), 10, draw_color, thickness)
            cv2.circle(canvas, (x + w // 2, y + h // 2), thickness, draw_color, -1)

    # Preprocess the canvas to pass to Tesseract OCR
    thresh_image = preprocess_for_ocr(canvas)

    # Initialize text variable
    text = ""

    # OCR to detect text with specific character whitelist for letters and numbers only
    try:
        text = pytesseract.image_to_string(thresh_image, config='--psm 7 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        print(text)  # Print output for debugging
    except KeyError as e:
        print(f"KeyError: {e}")  # Catch and print KeyError for debugging
    except pytesseract.pytesseract.TesseractError as e:
        print(f"Tesseract error: {e}")  # Catch Tesseract-specific errors
    except Exception as e:
        print(f"An error occurred: {e}")  # Catch any other exceptions

    # Display detected text
    cv2.putText(frame, f"Detected Text: {text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Combine the canvas with the video frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the combined image
    cv2.imshow("Air Canvas", combined)

    # Press 'c' to clear the canvas, 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
