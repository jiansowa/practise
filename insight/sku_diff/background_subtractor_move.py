import cv2
import numpy as np

#bs = cv2.createBackgroundSubtractorKNN(detectShadows = True)
#bs = cv2.createBackgroundSubtractorMOG2(detectShadows = True)
#default parameter (500,16,true)
bs = cv2.createBackgroundSubtractorMOG2(500,64,False)
#bs = cv2.createBackgroundSubtractorKNN()
#bs = cv2.createBackgroundSubtractorGMG(detectShadows = True)

#camera = cv2.VideoCapture('rtsp://192.168.159.135:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?')
camera = cv2.VideoCapture("/home/jiansowa/Videos/seg2.mp4")
#camera = cv2.VideoCapture("/home/jiansowa/samba_home/openpose/examples/media/manypeople.mp4")

while True:
    try:
        ret, frame = camera.read()
    except Exception as e:
        camera = cv2.VideoCapture('rtsp://192.168.159.135:554/user=test&password=aaaa1111&channel=1&stream=0.sdp?')
        print("Failed to grab", e)
        continue

    if frame is None:
        continue

    fgmask = bs.apply(frame)
    fgmask = cv2.GaussianBlur(fgmask, (21, 21), 0)
    #dilate(foreGround, foreGround, Mat(), Point(-1, -1), 3);
    #erode(foreGround, foreGround, Mat(), Point(-1, -1), 6);
    #dilate(foreGround, foreGround, Mat(), Point(-1, -1), 3);
    print(fgmask.shape)
    th = cv2.threshold(fgmask.copy(), 127, 255, cv2.THRESH_BINARY)[1]
    #dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations = 2)
    #dilated = cv2.dilate(th, None, iterations = 2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1600:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

    #cv2.imshow("mog", fgmask)
    cv2.imshow("thresh", th)
    #cv2.imshow("detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()

