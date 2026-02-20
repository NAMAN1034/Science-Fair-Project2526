#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_ICM20948.h>

Adafruit_ADS1115 emg;
Adafruit_ICM20948 imu;
//I named these emg and imu instead of the generic ads and icm bc I wanted to represent what the sensors were actually "grabbing" and it's easier to read in the code when I have to call the functions for these sensors

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10); //have to wait for port to connect
    Serial.println("sensor probe starting");
    Wire.begin();
    //finding emg ad8232(emg) sensor
    if (!emg.begin(0x48)) {
        Serial.println("ADS1115 not found");
    }   else {
        Serial.println("ADS1115 found");
        emg.setGain(GAIN_ONE); //amplified
    }
    //finding imu icm20948 sensor
    if (!imu.begin_I2C(0x68)) {
        Serial.println("ICM20948 not found");    
    }   else {
        Serial.println("ICM20948 found");
    }
}

void loop() {
    //read the emg sensor
    int16_t emg_value = emg.readADC_SingleEnded(0); //16 bit value is best fit for 8 bytes
    //read the imu sensor
    sensors_event_t accel, gyro, temp, mag;
    imu.getEvent(&accel, &gyro, &temp, &mag);

    //print values from both sensors to serial monitor
    Serial.print("EMG"); 
    Serial.print(emg_value);
    Serial.print(" | Gyro_X(Pitch):");
    Serial.print(gyro.gyro.x);
    Serial.print(" | Gyro_Y(Roll):");
    Serial.print(gyro.gyro.y);
    Serial.print(" | Gyro_Z(Yaw):");
    Serial.print(gyro.gyro.z);

    delay(100); //delay for readability
}