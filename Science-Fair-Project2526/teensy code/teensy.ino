#include <Wire.h>
#include <Adafruit_ADS1X15.h>
#include <Adafruit_ICM20948.h>
#include <Adafruit_Sensor.h>
#include <Arduino.h>
#include <string.h>
#include <math.h>

static constexpr uint32_t USB_BAUD = 200000; //super fast rate for faster comms with pi
static constexpr uint8_t FRAME_VERSION = 1; //version of the data frame being sent to pi, will be used for parsing on pi side which i'll take care of later
static constexpr uint8_t FRAME_SAMPLES = 8; //number of samples to send in each frame, this is a balance between sending enough data for the pi to work with and not overwhelming the serial connection. i'll tweak this later based on how the data looks when i start testing
static constexpr uint32_t EMG_FS_HZ = 860; //was gonna have it at 1000 but pi only allows up to 860 for the ads1115 so had to lower it still fast enough for emg tho
static constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / EMG_FS_HZ; //period between samples in ms
static constexpr uint32_t IMU_FS_HZ = 200; //slower than emg rate bc imu data is less time sensitive and this will help reduce the amount of data being sent to the pi to not overwhelm it
static constexpr uint32_t IMU_SAMPLE_PERIOD_US = 1000000UL / IMU_FS_HZ; //same thing as above but for imu

struct __attribute__((packed)) Sample4 {
    int16_t emg;
    int16_t gx;
    int16_t gy;
    int16_t gz; 
    //16 bit values are best fit for 8 bytes also the gyro values are small enough to fit in 16 bits so i can save space on pi by using int16_t instead of float
};

static Sample4 frame_buffer[FRAME_SAMPLES]; //created array to hold 8 samples per frame
static uint8_t frame_count = 0; //i count number of samples in current frame
static uint16_t frame_seq = 0; 
static uint64_t frame_t0_us =0; //time first sample taken
static uint64_t next_sample_us = 0; //deadline

static constexpr uint8_t PWM_PIN = 6; //pwm pin on teensy
static constexpr uint8_t PWM_RES_BITS = 12; //used 12 bit resolution for pwm to have more control over the duty cycle
static constexpr uint8_t ADS1115_ADDR = 0x48; //i2x address for emg
static constexpr uint8_t ICM20948_ADDR = 0x69; //i2c address for imu

static Adafruit_ADS1115 emg; //used emg instead of ads cuz it makes more sense to me to name it after the sensor rather than the adc it's using but i can change it later if i want
static Adafruit_ICM20948 imu; //same thing as above but for imu (icm was interchangeable)
static bool emg_ready = false;
static bool imu_ready = false;
static uint64_t last_emg_sample_us = 0;
static float emg_center = 0.0f; //avg baseline of emgs signal when relaxed
static bool emg_center_init = false;
static bool emg_have_sample = false;
static int16_t last_gx = 0;
static int16_t last_gy = 0;
static int16_t last_gz = 0;
static bool imu_have_sample = false;
static uint64_t next_imu_us = 0; //used to track when to take next imu sample based on imu sample period
static uint64_t last_emg_update_us = 0; //used to track when to update emg center based on how long it's been since last update
static uint64_t last_imu_update_us = 0; //same thing as above but for imu
static constexpr float EMG_CENTER_ALPHA = 0.0015f; //smoothing factor for emg avg calibration, it's low bc i want it to be very slow to change the center value so that it doesn't get thrown off by random spikes in the emg signal but it will still adapt over time if the baseline shifts due to things like electrode drying or slight changes in electrode placement
static constexpr float EMG_TO_12BIT_DIV = 8.0f; //used to convert emg value to 12 bit range for pwm output, the ads1115 has a range of -32768 to 32767 so dividing by 8 will give me a range of -4096 to 4095 which is close enough to the 0-4095 range of 12 bit pwm and will allow me to use the full range of the pwm output
static constexpr uint32_t EMG_STALE_US = 500000; //if it's been more than 500ms since last emg sample then consider the emg data stale and set it to center value to prevent erratic movements of the pwm output
static constexpr uint32_t IMU_STALE_US = 1000000; //same thing as above but for imu, set to 1s since imu data is less time sensitive than emg data

static constexpr float FREQ_MIN_HZ = 1.0f; //minimum freq for pwm output(*note to self* adjust accordingly later)
static constexpr float FREQ_MAX_HZ = 200.0f; //maximum freq for pwm output and also i'll change accordingly cuz rn it seems a bit high
static constexpr float PW_MIN_US = 10.0f; //minimum pulse width for pwm output (microseconds)
static constexpr float PW_MAX_US = 2000.0f; //maximum pulse width for pwm output (microseconds)
static constexpr float BURST_MIN_MS = 1.0f; //minimum burst duration for pwm output
static constexpr float BURST_MAX_MS = 200.0f; //maximum burst duration for pwm output

static constexpr uint32_t CMD_WATCHDOG_US = 200000; //if no cmds recieved for 200 ms then stop stim
static bool stim_on = false;
static float stim_freq_hz = 0.0f;
static float stim_pw_us = 0.0f;
static uint64_t stim_on_until_us = 0; //track when to off stim based on burst duration
static uint64_t last_cmd_us = 0;

static char cmd_line[96]; //buffer to hold incoming command line from pi, 96 chars should be enough for a simple command with some parameters but i can adjust later if needed
static uint8_t cmd_len = 0;

//func to help timing survive 32 bit micros rollover and not reset after ~70 minutes
uint64_t micros64() {
    static uint32_t last = 0;
    static uint64_t high = 0;
    const uint32_t now = micros();
    if (now < last) {
        high +=(1ULL << 32);
    }
    last = now;
    return high | static_cast<uint64_t>(now);
}

//for sending data frames to pi in correct format
template <typename T>
inline void writeLE(const T &v) {
    Serial.write(reinterpret_cast<const uint8_t *>(&v), sizeof(T));
}

//clamp functions to keep values within a range for overflow prevention
float clampf(float x, float lo, float hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

//clamp for signed 16 bit values, used for gyro data and converting emg to 12 bit range for pwm output
int16_t clampTo12Bit(int32_t x) {
    if (x<0) return 0;
    if (x>4095) return 4095;
    return static_cast<int16_t>(x);
}

int16_t clampToSigned16(float x) {
    if (x < -32768.0f) return -32768;
    if (x > 32767.0f) return 32767;
    return static_cast<int16_t>(x);
}

//convert annoying radians into degrees and sends ints
int16_t gyroRadToWireUnits(float rad_s) {
    const float deg_s = rad_s * 57.2957795f;
    return clampToSigned16(deg_s * 100.0f);
}

bool setupSensors() {
    Wire.begin();
    Wire.setClock(400000); //set i2c clock to 400kHz for rapid comms
    emg_ready = emg.begin(ADS1115_ADDR, &Wire);
    if (emg_ready) {
        emg.setGain(GAIN_ONE);
    }
    #ifdef RATE_ADS1115_860SPS
    emg.setDataRate(RATE_ADS1115_860SPS);
    #endif
    emg.startADCReading(ADS1X15_REG_CONFIG_MUX_SINGLE_0, true);

    imu_ready = imu.begin_I2C(ICM20948_ADDR, &Wire);
    return emg_ready && imu_ready;
}

void fatalSensorError(const char *msg) {
    Serial.println("Sensor error,");
    Serial.println(msg);
    while (true) {
        offStim();
        delay(1000);
    }
}

void offStim() {
    stim_on = false;
    stim_freq_hz = 0.0f;
    stim_pw_us = 0.0f;
    stim_on_until_us = 0;
    analogWrite(PWM_PIN, 0); //turn off pwm output
    digitalWriteFast(PWM_PIN, LOW);
}

void applyPWM(float freq_hz, float pw_us) {
    const float period_us = 1000000.0f/freq_hz;
    const float pw_limited = fminf(pw_us, 0.95f * period_us);
    float duty_cycle = (pw_limited / freq_hz)/1000000.0f; //duty cycle = pulse width / period
    duty_cycle = clampf(duty_cycle, 0.0f, 0.95f); //limit duty cycle to 95% to prevent issues with very long pulse widths
    analogWriteFrequency(PWM_PIN, freq_hz);
    const uint32_t max_counts =(1u<<PWM_RES_BITS)-1u;
    const uint32_t counts = static_cast<uint32_t>(duty_cycle * max_counts);
    analogWrite(PWM_PIN, counts);
}

void ackNow(const char *suffix = nullptr) {
    const u_int64_t t = micros64();//grab current time from clock
    Serial.print("ACK,");
    Serial.print(static_cast<uint32_t>(t & 0xFFFFFFFFu));//send bottom 32 bits
    if (suffix != nullptr) {
        Serial.print(",");
        Serial.print(suffix);
    }
    Serial.print("\n");
}

void applyStimCmd(float freq_hz, float pw_us, float burst_ms) {
    const uint64_t now = micros64();
    last_cmd_us = now;//pi is alive and talking

    if (freq_hz <= 0.0f || pw_us <= 0.0f || burst_ms <= 0.0f) {
        offStim();
        ackNow("OFF");
        return;
    }
    //limits
    freq_hz = clampf(freq_hz, FREQ_MIN_HZ, FREQ_MAX_HZ);
    pw_us = clampf(pw_us, PW_MIN_US, PW_MAX_US);
    burst_ms = clampf(burst_ms, BURST_MIN_MS, BURST_MAX_MS);

    stim_freq_hz = freq_hz;
    stim_pw_us = pw_us;
    stim_on_until_us = now + static_cast<uint64_t>(burst_ms * 1000.0f);
    stim_on = true;
    applyPWM(stim_freq_hz, stim_pw_us);//toggling pin 6
    ackNow("ON");
}

//gatekeeper
void processCmdLine() {
    if (strncmp(cmd_line, "STIM,", 5) != 0) {
        return;
    }
    //sections for stim
    float freq_hz = 0.0f;
    float pw_us = 0.0f;
    float burst_ms = 0.0f;
    const int n = sscanf(cmd_line + 5, "%f,%f,%f", &freq_hz, &pw_us, &burst_ms);    if (n==3) {
        applyStimCmd(freq_hz, pw_us, burst_ms);
    }   else {
        offStim();
        ackNow("ERR");
    }
}

//receptionist
void pollCmdSerial() {
    //teensy speed>usb data arrival
    while (Serial.available()>0) {
        const char c = static_cast<char>(Serial.read());
        //new line = end cmd
        if (c=='\n' || c=='\r') {
            if (cmd_len > 0) {
                cmd_line[cmd_len] = '\0';
                processCmdLine();
                cmd_len = 0;
            }
            continue;
        }
        //turn into single word
        if (cmd_len < sizeof(cmd_line)-1) {
            cmd_line[cmd_len++] = c;
        }else {
            cmd_len = 0;
            offStim();
            ackNow("CMD_OVF");
        }
    }
}

bool readEMGSample(uint64_t t_us, int16_t &emg_out) {
    if(!emg_ready) return false;
        //read latest emg values
        last_emg_sample_us = emg.getLastConversionResults();
        emg_have_sample = true;
        last_emg_update_us = t_us;

    if (!emg_have_sample) return false; //no updates in >50ms so probably not connected or something
    if (static_cast<int64_t>(t_us - last_emg_update_us)>static_cast<int64_t>(EMG_STALE_US)) {
        return false;
    }

    const float raw_f = static_cast<float>(last_emg_sample_us);
    if (!emg_center_init) {
        emg_center = raw_f;
        emg_center_init = true;
    }   else {
        emg_center = (1.0f-EMG_CENTER_ALPHA)*emg_center+EMG_CENTER_ALPHA*raw_f; //emg drift prevented by low pass filter
    }
    const float centered = raw_f - emg_center; //rest state=0
    const float emg_12f = 2048.0f+centered/EMG_TO_12BIT_DIV; //middle of 12 bit
    emg_out = clampTo12Bit(static_cast<int32_t>(lroundf(emg_12f))); //fit on graph
    return true;
}

bool readIMUSample(uint64_t t_us, int16_t &gx, int16_t &gy, int16_t &gz) {
    if (!imu_ready) return false;
    if (next_imu_us == 0) next_imu_us = t_us+IMU_SAMPLE_PERIOD_US; 
    //metronome
    if (static_cast<int64_t>(t_us-next_imu_us)>=0) {
        sensors_event_t accel, gyro, temp, mag;
        //pull raw rotation data from imu and convert to degrees
        if (imu.getEvent(&accel, &gyro, &temp, &mag)) {
            last_gx = gyroRadToWireUnits(gyro.gyro.x);
            last_gy = gyroRadToWireUnits(gyro.gyro.y);
            last_gz = gyroRadToWireUnits(gyro.gyro.z);
            imu_have_sample = true;
            last_imu_update_us = t_us;
        }
        next_imu_us += IMU_SAMPLE_PERIOD_US; 
        //drift prevention
        if (next_imu_us ==0 || static_cast<int64_t>(t_us-next_imu_us)>static_cast<int64_t>(IMU_SAMPLE_PERIOD_US*4)) {
            next_imu_us = t_us + IMU_SAMPLE_PERIOD_US;
        }
    }
    if (!imu_have_sample) return false;
    //frozen motion prevention
    if(static_cast<int64_t>(t_us-last_imu_update_us)>static_cast<int64_t>(IMU_STALE_US)) {
        return false;
    }
    gx = last_gx;
    gy = last_gy;
    gz = last_gz;
    return true;
}

void sendFrame(uint64_t t0_us, uint8_t n) {
    static const uint8_t hdr[3] = {'T', 'S', 'F'}; //look for tsf
    Serial.write(hdr, sizeof(hdr));
    Serial.write(FRAME_VERSION);    
    writeLE(frame_seq);
    writeLE(t0_us); //tremor occurence time
    Serial.write(n);
    Serial.write(reinterpret_cast<const uint8_t *>(frame_buffer), n * sizeof(Sample4)); //raw bytes not muscle data rn
    frame_seq++;
}

void streamSamples() {
    const uint64_t now = micros64();
    uint16_t produced = 0;
    //is it time for a sample?
    while (static_cast<int64_t>(now-next_sample_us)>=0) {
        Sample4 s{};
        int16_t gx = 0, gy = 0, gz = 0, emg = 0;
        const bool emg_ok = readEMGSample(next_sample_us, emg);
        const bool imu_ok = readIMUSample(next_sample_us, gx, gy, gz);
        //sensor check. if either sucks then just drop it
        if (emg_ok && imu_ok) {
            if (frame_count == 0) {
                frame_t0_us = next_sample_us;
            }
            s.emg = emg;
            s.gx = gx;
            s.gy = gy;
            s.gz = gz;
            frame_buffer[frame_count++] = s;
        }   else {
            frame_count = 0;
        }

        next_sample_us += SAMPLE_PERIOD_US;
        //ship 8 samples as one binary pack
        if (frame_count >= FRAME_SAMPLES) {
            sendFrame(frame_t0_us, frame_count);
            frame_count = 0;
        }
        if (produced > 200) {
            break;
        }
    }
}

void serviceStimSafety() {
    //teensy exit function to save energy and prevent issues if pi dies or something, if no stim just exit
    if (!stim_on) {
        return;
    }

    const uint64_t now = micros64();
    const bool burst_expired = static_cast<int64_t>(now-stim_on_until_us)>=0; //sharp 'pulses' instead of continuous
    const bool cmd_expired = static_cast<int64_t>(now-last_cmd_us)>=static_cast<int64_t>(CMD_WATCHDOG_US); //safety. stop zapping rn if no cmd received for a while

    if (burst_expired || cmd_expired) {
        offStim();
    }   else {
        applyPWM(stim_freq_hz, stim_pw_us);
    }
}

void setup() {
    analogWriteResolution(PWM_RES_BITS);
    pinMode(PWM_PIN, OUTPUT); //ouput for zaps
    offStim();//safety!!!!
    Serial.begin(USB_BAUD);
    delay(200); //added after teensy kept crashing when i tried comms to early
    //check if sensors are working
    if(!setupSensors()) {
        fatalSensorError("Sensors missing");
    }
    next_imu_us=micros64()+IMU_SAMPLE_PERIOD_US;
    //priming
    const u_int64_t prime_deadline = micros64()+2000000ULL; //warm up sensors for 2 secs
    //wait for green lights before 'takeoff'
    while(micros64()<prime_deadline) {
        int16_t emg=0, gx=0, gy=0, gz=0;
        const u_int64_t t_us = micros64();
        const bool emg_ok = readEMGSample(t_us, emg);
        const bool imu_ok = readIMUSample(t_us, gx, gy, gz);
        if (emg_ok && imu_ok) break;
        delay(1);
    }
}

void loop() {
    pollCmdSerial();
    streamSamples();
    serviceStimSafety();
}