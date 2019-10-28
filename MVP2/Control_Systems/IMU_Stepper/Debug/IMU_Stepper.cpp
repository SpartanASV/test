#include "Energia.h"

#line 1 "C:/Users/scott/OneDrive/Documents/SJSU/19B_FA/ENGR160SL/MC workspace/IMU_Stepper/IMU_Stepper.ino"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Stepper.h>
#include <Servo.h>


#define BNO055_SAMPLERATE_DELAY_MS (100)
void calibrate(void);
void setHeading(void);
void setup(void);
void loop(void);

#line 10
const int stepsPerRevolution = 100;  
                                    

 

Stepper myStepper(stepsPerRevolution, 34,35,36,37);

int stepCount = 0;         
double heading;

byte servoPin = 31; 
Servo servo;       



Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28);


void calibrate(void)
{
  while(1){

  
  
  
  
  
  
  
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

  
  Serial.print("X: ");
  Serial.print(euler.x());
  Serial.print(" Y: ");
  Serial.print(euler.y());
  Serial.print(" Z: ");
  Serial.print(euler.z());
  Serial.print("\t\t");

  uint8_t system, gyro, accel, mag = 0;
  bno.getCalibration(&system, &gyro, &accel, &mag);
  Serial.print("CALIBRATION: Sys=");
  Serial.print(system, DEC);
  Serial.print(" Gyro=");
  Serial.print(gyro, DEC);
  Serial.print(" Accel=");
  Serial.print(accel, DEC);
  Serial.print(" Mag=");
  Serial.println(mag, DEC);

  delay(BNO055_SAMPLERATE_DELAY_MS);

  if(system == 3)break;

  }
}


void setHeading(void){
    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

    double x = 0;

    for(int i = 1; i<101; i++){
        euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        x += euler.x();
    }

    heading = x/100;

}

void setup(void)
{
  Serial.begin(115200);
  Serial.println("Orientation Sensor Raw Data Test"); Serial.println("");

  
  if(!bno.begin())
  {
    
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }

  delay(1000);

  
  bno.setExtCrystalUse(true);

  
  Serial.println("Calibration status values: 0=uncalibrated, 3=fully calibrated");
  calibrate();
  Serial.println("Calibration completed.  heading will be set in 5 seconds");

  
  delay(5000);
  setHeading();
  Serial.print("Heading set to: x = ");
  Serial.println(heading);

  myStepper.setSpeed(100);

  servo.attach(servoPin);

  servo.writeMicroseconds(1500); 

  delay(7000); 
}

int previous = 0;
imu::Vector<3> euler;
int sig = 1200;
bool flag = 0;

void loop(void){

    euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    double correction = 0;

    if(euler.x() < heading-2 || euler.x()>heading+2){
        correction = heading-euler.x();
    }

    if(correction != 0){
        Serial.print("Correct: x by ");
        Serial.println(correction);
    }

    myStepper.step(correction - previous);

    
    previous = correction;




    sig = 1500;

    Serial.println(sig);

    servo.writeMicroseconds(sig); 
}



