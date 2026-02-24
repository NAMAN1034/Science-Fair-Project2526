from __future__ import annotations
import argparse
import sys
import time
import logging
import datetime
import serial
from serial.tools import list_ports

#logging setup
logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",datefmt="%H:%M:%S",)
log = logging.getLogger("TeensyCheck")
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="teensy finder")
    parser.add_argument("--port", type=str, default="", help="specific port)")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds to run)")
    parser.add_argument("--ping", type=float, default=2.0, dest="ping_interval", help="seconds between pings")
    parser.add_argument("--list", action="store_true", help="list the ports")
    return parser.parse_args()
def list_serial_ports() -> list:
    return list(list_ports.comports())
def pick_port(user_port: str) -> str:
    if user_port:
        return user_port
    ports = list_serial_ports()
    if not ports:
        raise ConnectionError("no serial ports")
    for port in ports:
        text = f"{port.description} {port.manufacturer}".lower()
        if "teensy" in text:
            log.info(f"found Teensy on {port.device}")
            return port.device
    for port in ports:
        name = port.device.lower()
        if "usb" in name or "acm" in name or "modem" in name:
            log.info(f"probably using usb serial device: {port.device} ({port.description})")
            return port.device
    log.warning(f"no known match. using {ports[0].device}")
    return ports[0].device
def run_diagnostic() -> int:
    args = get_args()
    if args.list:
        ports = list_serial_ports()
        print("\navailable serial ports")
        if not ports:
            print("no serial ports found")
            return 0
        for port in ports:
            print(f"{port.device:<15} | {port.description} [{port.manufacturer}]")
        return 0
    try:
        port_name = pick_port(args.port)
    except ConnectionError as err:
        log.error(err)
        return 1
    #i put these here so they always exist for the summary print
    rx_count = 0
    tx_count = 0
    try:
        with serial.Serial(port_name, args.baud, timeout=0.1) as ser:
            log.info(f"Oopened {port_name} at {args.baud} baud")
            time.sleep(1.5)  #teensy can reboot when the port opens
            ser.reset_input_buffer()
            #wakeup message
            ser.write(b"HELLO\n")
            log.info("sent the hello msg and starting a listening loop")
            start = time.monotonic()
            last_ping = 0.0
            while True:
                now = time.monotonic()
                elapsed = now - start
                if args.duration > 0 and elapsed >= args.duration:
                    log.info(f"finished after {args.duration} seconds")
                    break
                if args.ping_interval > 0 and (now - last_ping) >= args.ping_interval:
                    ser.write(b"PING\n")
                    tx_count += 1
                    last_ping = now
                if ser.in_waiting > 0:
                    line = ser.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        print(f"[Teensy] {line}")
                        rx_count += 1
                time.sleep(0.005)
    except serial.SerialException as err:
        log.error(f"Serial error: {err}")
        return 1
    except KeyboardInterrupt:
        log.info("stopped by me")
    #summary!!!
    print("summary")
    print(f"lines Received:{rx_count}")
    print(f"pings Sent:{tx_count}")
    print(f"status:{'SUCCESS' if rx_count > 0 else 'no data recieved'}")

    if rx_count == 0:
        print("\i gotta double check the if baud rate in teensy ino is same")
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(run_diagnostic())
