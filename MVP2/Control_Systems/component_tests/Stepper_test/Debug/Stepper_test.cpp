#include "Energia.h"

#line 1 "C:/Users/scott/OneDrive/Documents/SJSU/19B_FA/ENGR160SL/MC workspace/Stepper_test/Stepper_test.ino"




















#include <Stepper.h>

void setup();
void loop();

#line 23
const int stepsPerRevolution = 100;  
                                     



Stepper myStepper(stepsPerRevolution, 34,35,36,37);

int stepCount = 0;         

void setup() {
  
  Serial.begin(115200);
}

int count = 0;
void loop() {
  myStepper.setSpeed(100);

  
  Serial.print("stepping forward ");
  Serial.println(count);

  myStepper.step(100);

  

  myStepper.step(-100);
  



























}




