import numpy as np
import cv2

cap = cv2.VideoCapture(1)
ret, frame = cap.read()

while(True):
		cv2.imshow('img1',frame)
		if cv2.waitkey(1) & 0xFF == ord('y'):
				cv2.imwrite('images/c1.png',frame)
				cv2.destroyAllWindows()
				break
cap.release()
