#include "Energia.h"

#line 1 "C:/Users/scott/OneDrive/Documents/SJSU/19B_FA/ENGR160SL/MC workspace/IMU_adapted/IMU_adapted.ino"
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
















#define BNO055_SAMPLERATE_DELAY_MS (100)



void calibrate(void);
void setHeading(void);
void setup(void);
void loop(void);

#line 25
Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28);

double heading[3];

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
    double y = 0;
    double z = 0;

    for(int i = 1; i<101; i++){
        euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        x += euler.x();
        y += euler.y();
        z += euler.z();
    }

    heading[0] = x/100;
    heading[1] = y/100;
    heading[2] = z/100;

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

  
  int8_t temp = bno.getTemp();
  Serial.print("Current Temperature: ");
  Serial.print(temp);
  Serial.println(" C");
  Serial.println("");

  bno.setExtCrystalUse(true);

  Serial.println("Calibration status values: 0=uncalibrated, 3=fully calibrated");
  calibrate();
  Serial.println("Calibration completed.  heading will be set in 5 seconds");
  delay(5000);
  setHeading();
  Serial.print("Heading set to: x = ");
  Serial.print(heading[0]);
  Serial.print(" y = ");
  Serial.print(heading[1]);
  Serial.print(" z = ");
  Serial.println(heading[2]);
  delay(1000);
}








void loop(void){

    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    double correction[] = {0,0,0};

    if(euler.x() < heading[0]-5 || euler.x()>heading[0]+5){
        correction[0] = heading[0]-euler.x();
    }
    if(euler.y() < heading[1]-5 || euler.y() > heading[1]+5){
        correction[1] = heading[1] - euler.y();
    }
    if(euler.z() < heading[2]-5 || euler.z() > heading[2]+5){
        correction[2] = heading[2] - euler.z();
    }

    if(correction[0] != 0 || correction[1] != 0 || correction[2] != 0){
        Serial.print("Correct: x by ");
        Serial.print(correction[0]);
        Serial.print(" y by ");
        Serial.print(correction[1]);
        Serial.print(" z by ");
        Serial.println(correction[2]);
    }

    delay(1000);

}



