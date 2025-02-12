import cv2
import numpy as np
import os

# Create a directory for the dataset
dataset_path = "shapes_symbols_dataset"
os.makedirs(dataset_path, exist_ok=True)

shapes = ['circle', 'square', 'triangle', 'plus', 'minus', 'multiply', 'divide']
num_images = 100

for shape in shapes:
    for i in range(num_images):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White background
        if shape == 'circle':
            cv2.circle(img, (100, 100), 50, (0, 0, 255), -1)  # Red circle
        elif shape == 'square':
            cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)  # Green square
        elif shape == 'triangle':
            points = np.array([[100, 30], [30, 170], [170, 170]], np.int32)
            cv2.fillPoly(img, [points], (255, 0, 0))  # Blue triangle
        elif shape == 'plus':
            cv2.rectangle(img, (80, 40), (120, 160), (0, 0, 255), -1)  # Red vertical
            cv2.rectangle(img, (40, 80), (160, 120), (0, 0, 255), -1)  # Red horizontal
        elif shape == 'minus':
            cv2.rectangle(img, (40, 100), (160, 140), (0, 0, 255), -1)  # Red horizontal
        elif shape == 'multiply':
            cv2.line(img, (50, 50), (150, 150), (0, 0, 255), 5)  # Red diagonal
            cv2.line(img, (150, 50), (50, 150), (0, 0, 255), 5)  # Red diagonal
        elif shape == 'divide':
            cv2.rectangle(img, (80, 90), (120, 110), (0, 0, 255), -1)  # Red line
            cv2.circle(img, (100, 130), 10, (0, 0, 255), -1)  # Red dot below line
        cv2.imwrite(os.path.join(dataset_path, f"{shape}_{i}.png"), img)

print("Synthetic dataset generated.")
