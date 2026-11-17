import serial
import time

arduino = serial.Serial("COM11", 115200, timeout=1)

# Wait for Arduino Uno to restart
time.sleep(2)

def move(left, right, duration):
    start_time = time.time()

    while time.time() - start_time < duration:
        command = f"{left},{right}\n"
        arduino.write(command.encode())
        time.sleep(0.1)

def stop():
    arduino.write(b"0,0\n")

try:
    print("Forward")
    move(-200, -200, 3)
    
    # print("backward")
    # move(200, 200, 5)

    # print("Turn right")
    # move(-200, 200, 2)

    # print("Turn Left")
    # move(200, -200, 2)

    # print("Stopping")
    # stop()

finally:
    stop()
    time.sleep(0.1)
    arduino.close()