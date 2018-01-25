#!/usr/bin/env python
import numpy as np
import cv2

cap = cv2.VideoCapture('rtsp://192.168.159.135:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?')

#bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
bs = cv2.createBackgroundSubtractorKNN()
#bs = cv2.createBackgroundSubtractorMOG2()
#bs = cv2.createBackgroundSubtractorGMG()

print("before loop!")
while(1):
    print("enter loop")
    try:
        ret, frame = cap.read()
    except Exception as e:
        cap = cv2.VideoCapture('rtsp://192.168.159.135:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?')
        print("Failed to grab", e)
        continue
        #break
       
    if frame is None:
        continue

    #cv2.imshow("capture",frame)
    fgmask = bs.apply(frame)
    #cv2.imshow('fg',fgmask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("out loop")
cap.release()
cv2.destroyAllWindows()
