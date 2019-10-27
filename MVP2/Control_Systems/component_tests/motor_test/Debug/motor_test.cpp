#include "Energia.h"

#line 1 "C:/Users/scott/OneDrive/Documents/SJSU/19B_FA/ENGR160SL/MC workspace/motor_test/motor_test.ino"
#include <Servo.h>

void setup();
void loop();

#line 3
byte servoPin = 31;
Servo servo;

void setup() {
    servo.attach(servoPin);

    servo.writeMicroseconds(3000); 

    delay(7000); 
}

void loop() {
    int signal = 1700; 

    servo.writeMicroseconds(signal); 
}



