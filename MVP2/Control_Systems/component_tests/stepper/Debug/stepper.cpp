#include "Energia.h"

#line 1 "C:/Users/scott/OneDrive/Documents/SJSU/19B_FA/ENGR160SL/MC workspace/stepper/stepper.ino"




















#include <Stepper.h>

void setup();
void loop();

#line 23
const int stepsPerRevolution = 200;  
                                     



Stepper myStepper(stepsPerRevolution, 34,35,36,37);

int stepCount = 0;         

void setup() {
  
  Serial.begin(9600);
}

void loop() {
  
  myStepper.step(1);
  Serial.print("steps:" );
  Serial.println(stepCount);
  stepCount++;
  delay(500);
}




