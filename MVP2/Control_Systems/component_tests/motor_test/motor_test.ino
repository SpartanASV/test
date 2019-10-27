//the purpose of this file is to test the thruster to make sure you can connect to it
//before trying anything more complex

#include <Servo.h>

byte servoPin = 31;
Servo servo;

void setup() {
    servo.attach(servoPin);

    servo.writeMicroseconds(3000); // send "stop" signal to ESC.

    delay(7000); // delay to allow the ESC to recognize the stopped signal
}

void loop() {
    int signal = 1700; // Set signal value, which should be between 1100 and 1900

    servo.writeMicroseconds(signal); // Send signal to ESC.
}
