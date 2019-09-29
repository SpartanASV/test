import cv2

camera = cv2.VideoCapture(1)
for i in range(2):
    return_value, image = camera.read()

cv2.imwrite('testImage.png', image)

camera.release()


