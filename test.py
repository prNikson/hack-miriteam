# # from ultralytics import YOLO

# # model = YOLO('yolov8n.pt')

# # results = model.track(source='test2.mp4', show=True)

from qreader import QReader


import cv2
import numpy as np
cat_color_low = np.array([0, 26, 144])
cat_color_high = np.array([100, 255, 255])
cap = cv2.VideoCapture('test2.mp4')
qreader = QReader()
while True:
    img, frame = cap.read()
    original = frame.copy()
    cat_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(cat_hsv, cat_color_low, cat_color_high)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if(w > 90 and h > 90):
            ori = original[y:y+h, x:x+w]
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
            decoded_text = qreader.detect_and_decode(image=ori)
    cv2.imshow('original', original)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()