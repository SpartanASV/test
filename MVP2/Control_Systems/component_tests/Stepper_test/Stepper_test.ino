//The purpose of this file is to try different strategies for stepper control
/*
 Stepper Motor Control - one step at a time

 This program drives a unipolar or bipolar stepper motor.
 The motor is attached to digital pins 8 - 11 of the Arduino.

 The motor will step one step at a time, very slowly.  You can use this to
 test that you've got the four wires of your stepper wired to the correct
 pins. If wired correctly, all steps should be in the same direction.

 Use this also to count the number of steps per revolution of your motor,
 if you don't know it.  Then plug that number into the oneRevolution
 example to see if you got it right.

 Created 30 Nov. 2009
 by Tom Igoe

 */

#include <Stepper.h>

const int stepsPerRevolution = 100;  // change this to fit the number of steps per revolution
                                     // for your motor

// initialize the stepper library on pins 34 through 37:
//mapping to msp432: 34 = P2.3; 35 = P6.7; 36 = P6.6; 37 = P5.6;
Stepper myStepper(stepsPerRevolution, 34,35,36,37);
//int stepCount = 0;         // number of steps the motor has taken

void setup() {
  // initialize the serial port:
  Serial.begin(115200);
  myStepper.setSpeed(100);
}

int count = 0;
void loop() {

  // step one step:
  Serial.print("stepping forward ");
  Serial.println(count);

  myStepper.step(100);

  //delay(500);

  myStepper.step(-100);


  //delay(500);
//  for(int i = 0; i<10; i++){
//      myStepper.step(1);
//      delay(100);
//  }
//  Serial.println("stepping backward");
//  for(int i = 0; i<10; i++){
//        myStepper.step(-1);
//        delay(50);
//  }

//  Serial.println("stepping forward 2");
//  myStepper.step(2);
//  delay(1000);
//
//  Serial.println("stepping forward 4");
//  myStepper.step(4);
//  delay(1000);
//
//  Serial.println("stepping backward 2");
//  myStepper.step(-2);
//  delay(1000);
//
//  Serial.println("stepping backward 4");
//  myStepper.step(-4);
//  delay(1000);


}

