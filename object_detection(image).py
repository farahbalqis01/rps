# Import YOLO
from ultralytics import YOLO
# Import OpenCV
import cv2 as cv
import math

# Load pre-trained model
model = YOLO(model="content/runs/detect/train/weights/best.pt")
class_names = model.names

# Load image from 'image1.jpg' and store it as img
img = cv.imread('image1.jpg')
img = cv.resize(img, (1280, 720))

# Detect objects in the image and store them as results
results = model.predict(source=img)

# Draw the bounding boxes and labels on the image
# Under cv.putText, adjust the font size to 0.5
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put classname with confidence
        text = class_names[math.floor(box.cls)] + " " + str(math.floor(box.conf * 100)) + "%"
        cv.putText(img, text, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Show the image
cv.imshow('Image', img)
cv.waitKey(0)