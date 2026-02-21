#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_ICM20948.h>

Adafruit_ADS1115 emg; //create instance of emg sensor
Adafruit_ICM20948 imu; //create instance of imu sensor

unsigned long lastSampleTime = 0;
const int samplePeriod = 10; //100hz sample rate

void setup() {
    Serial.begin(115200);
    Wire.begin();

    if (!emg.begin()) {
        Serial.println("failed to find emg");
        while (1);
    }

    emg.setGain(GAIN_ONE);

    if(!imu.begin_I2C(0x69)) {
        Serial.println("failed to find imu");
        while (1);
    }
}

void loop() {
    if (millis()-lastSampleTime>=samplePeriod) {
        lastSampleTime = millis();

        //emg read a0 pin
        int16_t emgSample = emg.readADC_SingleEnded(0);
        //imu read gyro
        sensors_event_t accel, gyro, mag;
        imu.getEvent(&accel, &gyro, &mag);

        Serial.print(emgSample);
        Serial.print(",");
        Serial.print(accel.acceleration.x);
        Serial.print(",");
        Serial.print(accel.acceleration.y);
        Serial.print(",");
        Serial.print(accel.acceleration.z);
        Serial.print(",");
        Serial.print(gyro.gyro.x);
        Serial.print(",");
        Serial.print(gyro.gyro.y);
        Serial.print(",");
        Serial.println(gyro.gyro.z);
    }
}
