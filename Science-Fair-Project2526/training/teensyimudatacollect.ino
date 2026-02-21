#include <Adafruit_ICM20948.h>

Adafruit_ICM20948 icm; //create instance of imu sensor

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(10);

    if(!icm.begin_I2C()) {
        Serial.println("failed to find imu");
        while (1) { delay(10); }
    }
}

void loop() {
    sensors_event_t accel;
    sensors_event_t gyro;
    sensors_event_t mag;
    icm.getEvent(&accel, &gyro, &mag);

    Serial.print(accel.acceleration.x); Serial.print(",");
    Serial.print(accel.acceleration.y); Serial.print(",");
    Serial.print(accel.acceleration.z); Serial.print(",");
    Serial.print(gyro.gyro.x); Serial.print(",");
    Serial.print(gyro.gyro.y); Serial.print(",");
    Serial.println(gyro.gyro.z);

    delay(10); //100hz sample rate
}