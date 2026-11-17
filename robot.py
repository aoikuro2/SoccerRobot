
import cv2
import numpy as np
import serial
import time

# ==========================================
# SETTINGS
# ==========================================

CAMERA_INDEX = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SERIAL_PORT = "COM11"
BAUD_RATE = 115200

# Replace these with your FINAL calibrated HSV values.
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([15, 255, 255])

LOWER_RED2 = np.array([172, 120, 70])
UPPER_RED2 = np.array([179, 255, 255])

# Ball detection
MIN_BALL_AREA = 300

# Movement settings
FORWARD_SPEED = 65
MAX_TURN_SPEED = 90
MIN_TURN_SPEED = 35

CENTER_TOLERANCE = 45
KP = 0.35

# Tune this after testing the ball's apparent size.
STOP_AREA = 15000

# Motor direction correction.
# Change to -1 if a motor runs in the wrong direction.
LEFT_POLARITY = 1
RIGHT_POLARITY = 1

# ==========================================
# SERIAL CONNECTION
# ==========================================

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0.1
)

time.sleep(2)

def send_motor(left, right):
    left = int(np.clip(left, -255, 255))
    right = int(np.clip(right, -255, 255))

    left *= LEFT_POLARITY
    right *= RIGHT_POLARITY

    command = f"{left},{right}\n"
    arduino.write(command.encode())


def stop_robot():
    send_motor(0, 0)


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    arduino.close()
    raise SystemExit("Cannot open camera.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ==========================================
# BALL DETECTION
# ==========================================

def detect_ball(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(
        hsv, LOWER_RED1, UPPER_RED1
    )

    mask2 = cv2.inRange(
        hsv, LOWER_RED2, UPPER_RED2
    )

    mask = cv2.bitwise_or(mask1, mask2)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)

    if area < MIN_BALL_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(largest)

    center_x = x + w // 2
    center_y = y + h // 2

    ball = {
        "x": center_x,
        "y": center_y,
        "area": area,
        "box": (x, y, w, h)
    }

    return ball, mask


# ==========================================
# ROBOT MOVEMENT
# ==========================================

def follow_ball(ball):

    if ball is None:
        stop_robot()
        print("Ball not detected")
        return

    ball_x = ball["x"]
    ball_area = ball["area"]

    frame_center = FRAME_WIDTH // 2

    error = ball_x - frame_center

    # Ball is close enough: stop.
    if ball_area >= STOP_AREA:
        stop_robot()
        print("Ball reached - STOP")
        return

    # Ball is too far left or right:
    # Rotate in place to align with it.
    if abs(error) > CENTER_TOLERANCE:

        turn_speed = int(
            KP * abs(error)
        )

        turn_speed = np.clip(
            turn_speed,
            MIN_TURN_SPEED,
            MAX_TURN_SPEED
        )

        if error < 0:
            # Ball is on the left
            send_motor(-turn_speed, turn_speed)
            print("Turning LEFT")

        else:
            # Ball is on the right
            send_motor(turn_speed, -turn_speed)
            print("Turning RIGHT")

    else:
        # Ball is approximately centered
        send_motor(
            FORWARD_SPEED,
            FORWARD_SPEED
        )

        print("Moving FORWARD")


# ==========================================
# MAIN LOOP
# ==========================================

print("Robot ball-following started")
print("Press Q to stop")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            stop_robot()
            print("Camera error")
            break

        ball, mask = detect_ball(frame)

        # Draw camera center
        cv2.line(
            frame,
            (FRAME_WIDTH // 2, 0),
            (FRAME_WIDTH // 2, FRAME_HEIGHT),
            (255, 255, 0),
            2
        )

        if ball is not None:

            x, y, w, h = ball["box"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (ball["x"], ball["y"]),
                5,
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Area: {int(ball['area'])}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        follow_ball(ball)

        cv2.imshow("Robot Camera", frame)
        cv2.imshow("Ball Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    stop_robot()

    cap.release()
    cv2.destroyAllWindows()

    arduino.close()

    print("Robot stopped safely")
