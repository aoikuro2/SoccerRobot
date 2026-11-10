import time

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==========================================
# 0. SETTINGS (ubah di sini saja)
# ==========================================

CAMERA_INDEX = 3                      # /dev/video3
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MODEL_PATH = "efficientdet_lite0.tflite"
SCORE_THRESHOLD = 0.35

# Warna merah di HSV (hasil kalibrasi).
# Merah ada di dua ujung roda warna, jadi butuh dua rentang.
LOWER_RED1 = np.array([0, 131, 60])
UPPER_RED1 = np.array([7, 255, 255])
LOWER_RED2 = np.array([179, 131, 60])
UPPER_RED2 = np.array([179, 255, 255])

MIN_RED_RATIO = 0.08       # minimal persentase piksel merah di dalam kotak MediaPipe
MIN_AREA = 300             # luas minimal kontur untuk fallback OpenCV
MIN_CIRCULARITY = 0.35     # 1.0 = lingkaran sempurna
CENTER_DEADZONE = 40       # piksel; selisih kecil dari tengah dianggap "CENTER"

KERNEL = np.ones((7, 7), np.uint8)


# ==========================================
# 1. LOAD MEDIAPIPE MODEL
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

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Error: Cannot open camera!")
    detector.close()
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def make_red_mask(hsv_image):
    """Buat mask piksel merah (dua rentang digabung), lalu bersihkan noise."""
    mask1 = cv2.inRange(hsv_image, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv_image, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    # OPEN: hapus bintik kecil. CLOSE: tutup lubang kecil di badan bola.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    return mask


def find_ball_mediapipe(frame, mask, last_timestamp):
    """Cari 'sports ball' lewat MediaPipe, lalu pastikan ada piksel merah di dalamnya."""
    height, width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Timestamp harus selalu naik, kalau tidak MediaPipe akan error.
    timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp + 1)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    for detection in result.detections:
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

        x = int(x1 + moments["m10"] / moments["m00"])
        y = int(y1 + moments["m01"] / moments["m00"])
        radius = max(5, min(bbox.width, bbox.height) // 2)

        return (x, y, radius), timestamp_ms

    return None, timestamp_ms


def find_ball_opencv(mask):
    """Fallback: cari objek merah yang bentuknya mendekati lingkaran."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for contour in contours:
        # Convex hull menutup lekukan (jahitan, bagian mengkilap) sebelum dihitung
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
            best = (int(x), int(y), int(radius))
            best_area = area

    return best


def steering_text(ball_x, width):
    """Kasih tahu bola ada di kiri / kanan / tengah relatif ke kamera."""
    error = ball_x - width // 2

    if abs(error) <= CENTER_DEADZONE:
        return "CENTER", error
    if error < 0:
        return "LEFT", error
    return "RIGHT", error


# ==========================================
# 4. MAIN DETECTION LOOP
# ==========================================

print("Starting Soccer Ball Detection...")
print("Press Q to quit.")

last_timestamp = 0
last_printed = None

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Cannot receive camera frame.")
            break

        height, width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = make_red_mask(hsv)

        # 1) MediaPipe dulu, 2) kalau gagal pakai deteksi warna + bentuk
        ball, last_timestamp = find_ball_mediapipe(frame, mask, last_timestamp)
        source = "MediaPipe"

        if ball is None:
            ball = find_ball_opencv(mask)
            source = "OpenCV"

        # ==================================
        # 5. DISPLAY RESULTS
        # ==================================

        if ball is not None:
            ball_x, ball_y, ball_radius = ball
            direction, error = steering_text(ball_x, width)

            cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 255, 0), 3)
            cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"BALL FOUND ({source}): X={ball_x}, Y={ball_y}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"{direction} (offset {error:+d} px)",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Print hanya kalau arahnya berubah, supaya terminal tidak banjir
            if direction != last_printed:
                print(f"Ball: X={ball_x}, Y={ball_y}, direction={direction}")
                last_printed = direction

        else:
            cv2.putText(
                frame,
                "SEARCHING FOR BALL...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            if last_printed != "LOST":
                print("Ball lost")
                last_printed = "LOST"

        # Garis tengah kamera dan zona "CENTER"
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)
        cv2.line(
            frame,
            (width // 2 - CENTER_DEADZONE, 0),
            (width // 2 - CENTER_DEADZONE, height),
            (120, 120, 120),
            1,
        )
        cv2.line(
            frame,
            (width // 2 + CENTER_DEADZONE, 0),
            (width // 2 + CENTER_DEADZONE, height),
            (120, 120, 120),
            1,
        )

        cv2.imshow("Soccer Robot - Ball Detection", frame)
        cv2.imshow("Red Detection Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # ==================================
    # 6. CLEANUP
    # ==================================
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    print("Program stopped.")