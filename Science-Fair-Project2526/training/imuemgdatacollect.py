import serial
import csv
import pandas as pd
import time

#setup teensy connection to imu
arduino_port = "/dev/cu.usbmodem187570801"#for my mac to teensy
baud_rate = 115200 
run_time = 15 #seconds
output_file = "/Users/namanpradhan/scienceproject2526/Science-Fair-Project2526/training/imuandemgdata.csv"

#list for data
data = []

#connect to imu and teensy
print("starting connection to teensy...")
try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(3)#warm up time for teensy
    print("connection established")
except Exception as e:
    print("error connecting to teensy:", e)
    exit()

start_time = time.time()

#collect data from teensy
while (time.time()-start_time) < run_time:
    line = ser.readline().decode('utf-8').strip() #read stuff from teensy and decode it
    if line:
        values = line.split(',')
        try:
            clean_values = [float(v) for v in values]
            if len(clean_values) ==7: 
                current_time = time.time()
                clean_values.append(current_time) #add timestamps
                data.append(clean_values)
                
                if len(data) % 10==0:
                    print(f"reading row {len(data)}")

            

        except ValueError:
            pass #ignore any lines that can't be parsed into floats

#close connection to teensy
ser.close()
print("done")

#save data
print("saving data")
header = ["emg", "ax", "ay", "az", "gx", "gy", "gz", "time"]
if data:
    df = pd.DataFrame(data, columns=header)
    df.to_csv(output_file, index=False)
    print("data collected saved to", output_file)
else:
    print("no data collected")
