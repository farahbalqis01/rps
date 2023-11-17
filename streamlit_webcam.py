# Advanced Challenge 2
# Use streamlit to design a simple web app to perform object detection from picture captured from camera
from ultralytics import YOLO
import streamlit as st
import numpy as np
import cv2 as cv
import math

model = YOLO(model="content/runs/detect/train/weights/best.pt")
class_names = model.names

def detect_objects(img):
    results = model.predict(source=img)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put classname with confidence
            text = class_names[math.floor(box.cls)] + " " + str(math.floor(box.conf * 100)) + "%"
            cv.putText(img, text, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img

# Create a title and subheader in the app
st.title("Object Detection Web App")
st.write("This is a simple web app to detect objects in images")

# Create a file uploader widget
img_capture = st.camera_input("Take a picture of an animal")

# Check if picture has been taken
if img_capture is not None:
    # Convert the file to an opencv image.
    bytes_data = img_capture.getvalue()
    opencv_image = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR")

    # Apply object detection
    result_image = detect_objects(opencv_image.copy())

    # Display the output image
    st.image(result_image, channels="BGR")