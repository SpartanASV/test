import serial

ser = serial.Serial('/dev/ttyACM0')
# ser.baudrate = 115200
if(ser.is_open):
    print("serialis open!!!!")
print(ser.name)
x = ''
print(len(x))
while len(x)==0:
    print('inside: ')
    print(len(x))
    x = ser.read()

print('Got the message!!!')
 
ser.close()

import cv2
video_capture = cv2.VideoCapture(1)
# Check success
if not video_capture.isOpenend():
		raise Exception("could not open video device")
# Read picture. ret == True on success
ret, frame = video_capture.read()
# close device
video_capture.release()
