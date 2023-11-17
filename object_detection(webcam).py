# Import YOLO from ultralytics
from ultralytics import YOLO
# Import OpenCV
import cv2 as cv
import math

# Load rock-paper-scissors model
model = YOLO(model="content/runs/detect/train/weights/best.pt")
# Load the model's labels and store them as class_names
class_names = model.names

# Capture video from webcam
webcam = cv.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = webcam.read()
    # Quit when no more frame
    if not ret:
        print("Video Ended")
        break

    # Run object detector
    results = model.predict(source=frame)

    # Draw bounding boxes
    # Under cv.putText, adjust the font size to 0.7
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put classname with confidence
            text = class_names[math.floor(box.cls)] + " " + str(math.floor(box.conf * 100)) + "%"
            cv.putText(frame, text, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv.imshow('frame', frame)
    # stop when Q is pressed
    if cv.waitKey(25) == ord("q"):
         break

# When everything done, release the capture
webcam.release()
cv.destroyAllWindows()    