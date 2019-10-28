import numpy as np
import cv2
print("CV2 Version: ", cv2.getVersionString())


cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    (contours, hierarchy) = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # looping for contours
    for c in contours:
        if cv2.contourArea(c) < 10000:# 10000:
            continue
        # get bounding box from contour
        (x, y, w, h) = cv2.boundingRect(c)
        # draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = None

        for i in range(2):
            (_, img) = cap.read()

        cv2.imwrite('test.png', img)

    # cv2.imshow('foreground and background', fgmask)
    cv2.imshow('rgb', frame)
    # Exit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
