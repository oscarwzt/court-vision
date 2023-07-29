import cv2
from ultralytics import YOLO
import numpy as np
import cvzone
import RPi.GPIO as GPIO

model = YOLO("yolov8n.pt")

# setup for servo
servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
p = GPIO.PWM(servoPIN, 50)
p.start(7.5)

# initialize old_x
old_x = None

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    
    # if the frame was not successfully read, break the loop
    if not success:
        break
    
    results = model(img, stream = True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if model.names[int(box.cls)] == "person":
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                confidence = np.round(box.conf[0].cpu(), 3)
                cvzone.putTextRect(img, f'{model.names[int(box.cls)]} {confidence:.3f}', (x1, max(35, y1)),  thickness = 2)
                
                # get the center x-coordinate of the bounding box
                center_x = x1 + ((x2 - x1) / 2)
                
                # if the old_x is None (initial condition), assign the center_x to old_x
                if old_x is None:
                    old_x = center_x
                
                # check if the person has moved significantly
                if abs(center_x - old_x) > 50:  # adjust the value as per your requirement
                    old_x = center_x
                    
                    # move the servo based on center_x
                    duty_cycle = 2.5 + (center_x / img.shape[1]) * (12.5 - 2.5)
                    p.ChangeDutyCycle(duty_cycle)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
p.stop()
GPIO.cleanup()
