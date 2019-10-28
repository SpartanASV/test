#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Stepper.h>
#include <Servo.h>

// Set the delay between fresh samples
#define BNO055_SAMPLERATE_DELAY_MS (100)
const int stepsPerRevolution = 100;  // change this to fit the number of steps per revolution
                                    // for your motor

 // initialize the stepper library on pins 34 through 37:
//  mapping to msp432: 34 = P2.3; 35 = P6.7; 36 = P6.6; 37 = P5.6;
Stepper myStepper(stepsPerRevolution, 34,35,36,37);

int stepCount = 0;         // number of steps the motor has taken
double heading;

byte servoPin = 31; //pin that thruster esc is connected to
Servo servo;       // esc is controlled as a servo

 // Check I2C device address and correct line below (id, address)
//                                                   by default address is 0x29 or 0x28
Adafruit_BNO055 bno = Adafruit_BNO055(-1, 0x28);

//calibrate the IMU on startup.
void calibrate(void)
{
  while(1){

       // Possible vector values can be:
      //  - VECTOR_ACCELEROMETER - m/s^2
     //   - VECTOR_MAGNETOMETER  - uT
    //    - VECTOR_GYROSCOPE     - rad/s
   //     - VECTOR_EULER         - degrees
  //      - VECTOR_LINEARACCEL   - m/s^2
 //       - VECTOR_GRAVITY       - m/s^2
  imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

  // Display the floating point data 
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

//initialize the heading
void setHeading(void){
    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

    double x = 0;

    //produce the sum of 100 consecutive X-axis readings
    for(int i = 1; i<101; i++){
        euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        x += euler.x();
    }

    //divide by 100 to find the average
    heading = x/100;

}

void setup(void)
{
  Serial.begin(115200);
  Serial.println("Orientation Sensor Raw Data Test"); Serial.println("");

  // Initialise the IMU
  if(!bno.begin())
  {
    // There was a problem detecting the BNO055 ... check your connections
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }

  delay(500);

  //do a thing that Scott doesn't know what it does �\_(:|)_/�
  bno.setExtCrystalUse(true);

  //calibrate IMU
  Serial.println("Calibration status values: 0=uncalibrated, 3=fully calibrated");
  calibrate();
  Serial.println("Calibration completed.  heading will be set in 5 seconds");

  //give the user 5 seconds before setting the heading so they can put it in place
  delay(5000);
  setHeading();
  Serial.print("Heading set to: x = ");
  Serial.println(heading);

  myStepper.setSpeed(100);

  //associate servoPin with servo object
  servo.attach(servoPin);
  servo.writeMicroseconds(1500); // send "stop" signal to ESC.

  // delay to allow the ESC to recognize the stopped signal
  delay(7000);

}

imu::Vector<3> euler;
int threshold = 2;
int previous = 0;
int sig = 1200;
bool flag = 0;

void loop(void){

    //Read IMU sensor at the top of each loop
    euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    double correction = 0;

    //if the IMU has changed within a threshold
    if(euler.x() < heading-threshold || euler.x()>heading+threshold){
        correction = heading-euler.x();
    }

    if(correction != 0){
        Serial.print("Correct: x by ");
        Serial.println(correction);
    }

    myStepper.step(correction - previous);

    // remember the previous value of the sensor
    previous = correction;

  //    flag = sig<1200?0:flag;
 //     flag = sig>1800?1:flag;
//      sig = flag?sig-1:sig+1;
    sig = 1500;

    Serial.println(sig);

    servo.writeMicroseconds(sig); // Send signal to ESC.
}
