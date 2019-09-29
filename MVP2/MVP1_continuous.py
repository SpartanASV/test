import serial
import cv2
import time 
import datetime

# open serial; baudrate = 115200
ser = serial.Serial('/dev/ttyACM0')

# check if serial is open
if(ser.is_open):
    print("serial is open!!!!")
else:
    print("ERROR SERIAL NOT OPENED")
print(ser.name)

#open camera
camera = cv2.VideoCapture(1)

#check if camera is open
if camera.isOpened():
    print("camera is open!!!!")
else:
    print("ERROR CAMERA NOT OPENED")

# params for the putText() function in cv2
org = (400,450)
font = cv2.FONT_HERSHEY_COMPLEX
fontScale = .5
color = (255,255,255)
thickness = 1
line = cv2.LINE_AA
bottomL = True

# trigger is a string that will be filled by the serial message from arduino
trigger = ''

while True:
    message = ser.read()
    trigger = trigger+message

    # action is initiated by 10 char trigger because Arduino is reading so fast that it registers multiple inputs for a brief break in threshold. May want to add wait statement to arduino.
    if len(trigger) >9:
        #initialize time stamp string
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%Y_%H-%M-%S')

        # must take 3 images because first two are unusable; looks like opencv is capturing faster than camera is able to wake up.
        for i in range(2):
            return_value, imageRaw = camera.read()

        #putText() returns an image with the text specified in the function args.  Look at documentation for info on args.
        image = cv2.putText(imageRaw, st, org, font, fontScale, color, thickness)
        cv2.imwrite('testImage'+st+'.png', image)

        print('Image Captured!')
        time.sleep(1)
        trigger = ''
 
ser.close()
camera.release()

