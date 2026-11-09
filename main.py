
import cv2
import numpy as np
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 1. LOAD MEDIAPIPE MODEL
# ==========================================

MODEL_PATH = "efficientdet_lite0.tflite"

BaseOptions = python.BaseOptions

options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    score_threshold=0.35,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.ObjectDetector.create_from_options(options)


# ==========================================
# 2. CAMERA SETUP
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera!")
    exit()

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ==========================================
# 3. ORANGE COLOR RANGE (HSV)
# ==========================================

# Adjust these values depending on your ball
LOWER_ORANGE = np.array([5, 100, 100])
UPPER_ORANGE = np.array([25, 255, 255])


# ==========================================
# 4. MAIN DETECTION LOOP
# ==========================================

print("Starting Soccer Ball Detection...")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot receive camera frame.")
        break

    height, width = frame.shape[:2]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create orange color mask
    mask = cv2.inRange(
        hsv,
        LOWER_ORANGE,
        UPPER_ORANGE
    )

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ======================================
    # MEDIAPIPE OBJECT DETECTION
    # ======================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = time.monotonic_ns() // 1_000_000

    detection_result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    ball_found = False
    ball_x = 0
    ball_y = 0
    ball_radius = 0

    # ======================================
    # 5. FIND ORANGE SPORTS BALL
    # ======================================

    # First, check MediaPipe sports ball detections
    for detection in detection_result.detections:

        category = detection.categories[0]

        if category.category_name.lower() != "sports ball":
            continue

        bbox = detection.bounding_box

        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)

        x2 = min(width, x1 + bbox.width)
        y2 = min(height, y1 + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        # Check orange pixels inside the detected object
        ball_mask = mask[y1:y2, x1:x2]

        orange_pixels = cv2.countNonZero(ball_mask)

        total_pixels = ball_mask.size

        orange_ratio = orange_pixels / max(
            total_pixels, 1
        )

        # Require some orange pixels
        if orange_ratio > 0.08:

            M = cv2.moments(ball_mask)

            if M["m00"] > 0:

                ball_x = int(
                    x1 + M["m10"] / M["m00"]
                )

                ball_y = int(
                    y1 + M["m01"] / M["m00"]
                )

                ball_radius = max(
                    5, min(bbox.width, bbox.height) // 2
                )

                ball_found = True
                break

    # ======================================
    # 6. OPENCV FALLBACK
    # ======================================

    # If MediaPipe does not recognize the ball,
    # search for orange circular objects directly.

    if not ball_found:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_area = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 300:
                continue

            perimeter = cv2.arcLength(
                contour, True
            )

            if perimeter == 0:
                continue

            circularity = (
                4 * np.pi * area /
                (perimeter * perimeter)
            )

            # Filter out very irregular shapes
            if circularity < 0.35:
                continue

            (x, y), radius = cv2.minEnclosingCircle(
                contour
            )

            if area > best_area:

                best_area = area

                ball_x = int(x)
                ball_y = int(y)
                ball_radius = int(radius)

                ball_found = True

    # ======================================
    # 7. DISPLAY RESULTS
    # ======================================

    if ball_found:

        # Draw circle around ball
        cv2.circle(
            frame,
            (ball_x, ball_y),
            ball_radius,
            (0, 255, 0),
            3
        )

        # Draw center point
        cv2.circle(
            frame,
            (ball_x, ball_y),
            5,
            (0, 0, 255),
            -1
        )

        # Display coordinates
        cv2.putText(
            frame,
            f"BALL FOUND: X={ball_x}, Y={ball_y}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(
            f"Ball Position: X={ball_x}, Y={ball_y}"
        )

    else:

        cv2.putText(
            frame,
            "SEARCHING FOR BALL...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Draw camera center
    cv2.line(
        frame,
        (width // 2, 0),
        (width // 2, height),
        (255, 0, 0),
        1
    )

    # Show camera image
    cv2.imshow(
        "Soccer Robot - Ball Detection",
        frame
    )

    # Show orange mask for debugging
    cv2.imshow(
        "Orange Detection Mask",
        mask
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================
# 8. CLEANUP
# ==========================================

cap.release()
detector.close()
cv2.destroyAllWindows()

print("Program stopped.")
