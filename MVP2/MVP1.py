import serial
import cv2

ser = serial.Serial('/dev/ttyACM0')
camera = cv2.VideoCapture(1)
# ser.baudrate = 115200
if(ser.is_open):
    print("serial is open!!!!")
else:
    print("ERROR SERIAL NOT OPENED")
print(ser.name)

if camera.isOpened():
    print("camera is open!!!!")
else:
    print("ERROR CAMERA NOT OPENED")

x = ''
print(len(x))
while len(x)==0:
    print('inside: ')
    print(len(x))
    x = ser.read()
    if len(x) >0:
        for i in range(2):
            return_value, image = camera.read()

        cv2.imwrite('testImage.png', image)

print('Image Captured!')
 
ser.close()
camera.release()


