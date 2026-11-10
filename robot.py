import time

import cv2
import numpy as np
import serial
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 0. SETTINGS
# ==========================================

# Camera settings (the supplied detector uses Linux /dev/video3).
CAMERA_INDEX = 3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
USE_V4L2 = True

MODEL_PATH = "efficientdet_lite0.tflite"
SCORE_THRESHOLD = 0.35

# Arduino serial settings.
# Raspberry Pi examples: /dev/ttyACM0 or /dev/ttyUSB0
# Windows example: COM11
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

# Keep False for the first tests. Set True only after checking the robot
# with its wheels lifted and a physical battery disconnect within reach.
MOTORS_ENABLED = False

# Motor speeds: start conservatively and tune after testing.
SEARCH_SPEED = 75
TURN_SPEED = 100
APPROACH_SPEED = 110

# Ball positioning and stopping.
CENTER_DEADZONE = 40
STOP_RADIUS = 55

# Control loop settings.
COMMAND_INTERVAL = 0.10
LOST_FRAMES_BEFORE_SEARCH = 4

# Red ball HSV ranges (from the user's calibrated detector).
LOWER_RED1 = np.array([0, 131, 60])
UPPER_RED1 = np.array([7, 255, 255])
LOWER_RED2 = np.array([179, 131, 60])
UPPER_RED2 = np.array([179, 255, 255])

MIN_RED_RATIO = 0.08
MIN_AREA = 300
MIN_CIRCULARITY = 0.35
KERNEL = np.ones((7, 7), np.uint8)


# ==========================================
# 1. MEDIAPIPE MODEL
# ==========================================

options = vision.ObjectDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    score_threshold=SCORE_THRESHOLD,
    running_mode=vision.RunningMode.VIDEO,
)

detector = vision.ObjectDetector.create_from_options(options)


# ==========================================
# 2. CAMERA SETUP
# ==========================================

if USE_V4L2:
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    detector.close()
    raise SystemExit(
        f"Cannot open camera index {CAMERA_INDEX}. "
        "Check CAMERA_INDEX and USE_V4L2."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


# ==========================================
# 3. ARDUINO SERIAL SETUP
# ==========================================

try:
    arduino = serial.Serial(
        SERIAL_PORT,
        BAUD_RATE,
        timeout=1,
        write_timeout=1,
    )
    # Arduino Uno may reset when the serial port is opened.
    time.sleep(2)
    arduino.reset_input_buffer()
    print(f"Arduino connected on {SERIAL_PORT}")

except serial.SerialException as error:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    raise SystemExit(f"Could not connect to Arduino: {error}")


# ==========================================
# 4. BALL DETECTION FUNCTIONS
# ==========================================

def make_red_mask(hsv_image):
    """Create and clean a mask containing the calibrated red color."""
    mask1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, KERNEL, iterations=2
    )
    return mask


def find_ball_mediapipe(frame, mask, last_timestamp):
    """Find a MediaPipe sports-ball candidate and validate its red pixels."""
    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    timestamp_ms = max(
        time.monotonic_ns() // 1_000_000,
        last_timestamp + 1,
    )
    result = detector.detect_for_video(mp_image, timestamp_ms)

    for detection in result.detections:
        if not detection.categories:
            continue

        category = detection.categories[0]
        if category.category_name.lower() != "sports ball":
            continue

        bbox = detection.bounding_box

        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(width, bbox.origin_x + bbox.width)
        y2 = min(height, bbox.origin_y + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        ball_mask = mask[y1:y2, x1:x2]
        red_ratio = cv2.countNonZero(ball_mask) / max(ball_mask.size, 1)

        if red_ratio <= MIN_RED_RATIO:
            continue

        moments = cv2.moments(ball_mask)
        if moments["m00"] <= 0:
            continue

        ball_x = int(x1 + moments["m10"] / moments["m00"])
        ball_y = int(y1 + moments["m01"] / moments["m00"])
        ball_radius = max(5, min(bbox.width, bbox.height) // 2)

        return (ball_x, ball_y, ball_radius), timestamp_ms

    return None, timestamp_ms


def find_ball_opencv(mask):
    """Fallback: find the largest sufficiently circular red contour."""
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    best_ball = None
    best_area = 0

    for contour in contours:
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)

        if area < MIN_AREA:
            continue

        perimeter = cv2.arcLength(hull, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue

        if area > best_area:
            (x, y), radius = cv2.minEnclosingCircle(hull)
            best_ball = (int(x), int(y), int(radius))
            best_area = area

    return best_ball


# ==========================================
# 5. AUTONOMOUS MOVEMENT
# ==========================================

def decide_movement(ball, camera_width):
    """
    Return state and motor speeds.

    Motor direction convention from the robot's previous hardware tests:
      (-speed, -speed) = forward
      ( speed,  speed) = backward
      ( speed,-speed) = turn left
      (-speed, speed) = turn right
    """
    if ball is None:
        return "SEARCH", SEARCH_SPEED, -SEARCH_SPEED

    ball_x, ball_y, ball_radius = ball
    error_x = ball_x - (camera_width // 2)

    # Stop when the ball is close enough.
    if ball_radius >= STOP_RADIUS:
        return "STOP", 0, 0

    # Turn toward the ball if it is outside the center deadzone.
    if error_x < -CENTER_DEADZONE:
        return "ALIGN LEFT", TURN_SPEED, -TURN_SPEED

    if error_x > CENTER_DEADZONE:
        return "ALIGN RIGHT", -TURN_SPEED, TURN_SPEED

    # Ball is centered and not yet close: move forward.
    return "APPROACH", -APPROACH_SPEED, -APPROACH_SPEED


# ==========================================
# 6. ARDUINO MOTOR COMMUNICATION
# ==========================================

def send_motor_command(left_speed, right_speed):
    """Send 'left,right\\n' to the Arduino, clamped to -255..255."""
    if not MOTORS_ENABLED:
        left_speed = 0
        right_speed = 0

    left_speed = max(-255, min(255, int(left_speed)))
    right_speed = max(-255, min(255, int(right_speed)))

    command = f"{left_speed},{right_speed}\n"
    arduino.write(command.encode("ascii"))


def stop_motors():
    """Send an immediate stop command."""
    try:
        arduino.write(b"0,0\n")
    except (serial.SerialException, OSError):
        pass


# ==========================================
# 7. MAIN AUTONOMOUS LOOP
# ==========================================

print("========================================")
print(" AUTONOMOUS SOCCER ROBOT")
print("========================================")
print(f"Camera index: {CAMERA_INDEX}")
print(f"Arduino port: {SERIAL_PORT}")
print(f"Motor control enabled: {MOTORS_ENABLED}")
print("Press Q or ESC to quit.")

last_timestamp = 0
last_command_time = 0.0
missed_frames = 0
last_state = None
running = True

try:
    stop_motors()

    while running:
        ret, frame = cap.read()

        if not ret:
            print("Cannot receive camera frame.")
            break

        height, width = frame.shape[:2]

        # Run the user's calibrated red-ball detector.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = make_red_mask(hsv)

        ball, last_timestamp = find_ball_mediapipe(
            frame, mask, last_timestamp
        )
        detection_source = "MediaPipe"

        if ball is None:
            ball = find_ball_opencv(mask)
            detection_source = "OpenCV"

        if ball is not None:
            missed_frames = 0
            ball_x, ball_y, ball_radius = ball
            state, left_speed, right_speed = decide_movement(
                ball, width
            )
            error_x = ball_x - (width // 2)

        else:
            missed_frames += 1
            ball_x = ball_y = ball_radius = 0
            error_x = 0

            # Briefly stop during short detection interruptions.
            # Start searching only after several consecutive misses.
            if missed_frames < LOST_FRAMES_BEFORE_SEARCH:
                state = "BALL LOST - WAITING"
                left_speed = 0
                right_speed = 0
            else:
                state, left_speed, right_speed = decide_movement(
                    None, width
                )

        # Send motor commands at a fixed interval.
        current_time = time.monotonic()
        if current_time - last_command_time >= COMMAND_INTERVAL:
            send_motor_command(left_speed, right_speed)
            last_command_time = current_time

        # Print state changes rather than flooding the terminal.
        if state != last_state:
            print(
                f"State: {state} | "
                f"Left: {left_speed} | Right: {right_speed}"
            )
            last_state = state

        # Draw camera center and alignment boundaries.
        center_x = width // 2
        cv2.line(
            frame, (center_x, 0), (center_x, height),
            (255, 0, 0), 1
        )
        cv2.line(
            frame,
            (center_x - CENTER_DEADZONE, 0),
            (center_x - CENTER_DEADZONE, height),
            (120, 120, 120), 1
        )
        cv2.line(
            frame,
            (center_x + CENTER_DEADZONE, 0),
            (center_x + CENTER_DEADZONE, height),
            (120, 120, 120), 1
        )

        # Draw detection information.
        if ball is not None:
            cv2.circle(
                frame, (ball_x, ball_y), ball_radius,
                (0, 255, 0), 3
            )
            cv2.circle(
                frame, (ball_x, ball_y), 5,
                (0, 0, 255), -1
            )

            cv2.putText(
                frame,
                f"BALL ({detection_source}): X={ball_x}, Y={ball_y}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2,
            )
            cv2.putText(
                frame,
                f"ERROR: {error_x:+d}  RADIUS: {ball_radius}",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                frame,
                "SEARCHING FOR BALL...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 2,
            )

        cv2.putText(
            frame,
            f"STATE: {state}",
            (10, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 0), 2,
        )
        cv2.putText(
            frame,
            f"LEFT: {left_speed} RIGHT: {right_speed}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2,
        )

        motor_status = (
            "MOTORS ENABLED" if MOTORS_ENABLED
            else "MOTORS DISABLED"
        )
        cv2.putText(
            frame,
            motor_status,
            (width - 220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0) if MOTORS_ENABLED else (0, 0, 255),
            2,
        )

        cv2.imshow("Autonomous Soccer Robot", frame)
        cv2.imshow("Red Detection Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            running = False

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Always request zero motor speed before closing the connection.
    stop_motors()
    time.sleep(0.1)

    try:
        arduino.close()
    except (serial.SerialException, OSError):
        pass

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Robot controller stopped.")
