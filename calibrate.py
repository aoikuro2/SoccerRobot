import os
os.environ["QT_QPA_PLATFORM"] = "xcb"   # paksa XWayland supaya ukuran jendela dihormati

import cv2
import numpy as np

# ==========================================
# SETTINGS
# ==========================================
CAMERA_INDEX = 1      # samakan dengan CAMERA_INDEX di robot.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Nilai awal kontrol kamera (Logitech C922)
INIT_EXPOSURE = 150       # satuan 100 µs, rentang 3-2047
INIT_GAIN = 64            # 0-255
INIT_WB = 46              # satuan 100 K -> 4600 K, rentang 20-65
INIT_FOCUS = 30           # 0-255

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit(f"Tidak bisa membuka kamera index {CAMERA_INDEX}.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Matikan mode otomatis dulu (urutan penting: auto off, baru set nilainya)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual (V4L2), 3 = auto
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

win = "Kalibrasi"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, 700, 650)


def nothing(x):
    pass


# (nama, nilai awal, maksimum)
sliders = [
    # --- Rentang warna HSV ---
    ("H1 min", 0, 179), ("H1 max", 15, 179),     # merah-oranye (awal roda warna)
    ("H2 min", 172, 179), ("H2 max", 179, 179),  # merah (ujung roda warna)
    ("S min", 120, 255),                         # seberapa "pekat" warnanya
    ("V min", 70, 255),                          # seberapa terang
    # --- Kontrol kamera (dikunci, bukan auto) ---
    ("Exposure", INIT_EXPOSURE, 600),
    ("Gain", INIT_GAIN, 255),
    ("WB x100K", INIT_WB, 65),
    ("Focus", INIT_FOCUS, 255),
]
for name, val, mx in sliders:
    cv2.createTrackbar(name, win, val, mx, nothing)

g = lambda n: cv2.getTrackbarPos(n, win)

# Simpan nilai kamera terakhir; properti hanya dikirim ke kamera saat slider berubah
cam_last = {}


def apply_camera_settings():
    exposure = max(3, g("Exposure"))
    gain = g("Gain")
    wb = max(20, g("WB x100K")) * 100
    focus = g("Focus")

    if cam_last.get("exposure") != exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        cam_last["exposure"] = exposure
    if cam_last.get("gain") != gain:
        cap.set(cv2.CAP_PROP_GAIN, gain)
        cam_last["gain"] = gain
    if cam_last.get("wb") != wb:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        cam_last["wb"] = wb
    if cam_last.get("focus") != focus:
        cap.set(cv2.CAP_PROP_FOCUS, focus)
        cam_last["focus"] = focus
    return exposure, gain, wb, focus


# Klik di jendela "Kamera" untuk melihat nilai HSV di titik itu
state = {}


def pick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and "hsv" in param:
        h, s, v = param["hsv"][y, x]
        print(f"HSV di ({x},{y}): H={h} S={s} V={v}")


cv2.namedWindow("Kamera")
cv2.setMouseCallback("Kamera", pick, state)

print("P = cetak nilai | klik di jendela Kamera = lihat HSV titik | Q = keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    exposure, gain, wb, focus = apply_camera_settings()

    h1_min, h1_max = g("H1 min"), g("H1 max")
    h2_min, h2_max = g("H2 min"), g("H2 max")
    s_min, v_min = g("S min"), g("V min")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    state["hsv"] = hsv

    m1 = cv2.inRange(hsv, (h1_min, s_min, v_min), (h1_max, 255, 255))
    m2 = cv2.inRange(hsv, (h2_min, s_min, v_min), (h2_max, 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    hasil = cv2.bitwise_and(frame, frame, mask=mask)

    # Panel info di jendela kalibrasi (supaya nilai slider selalu terlihat)
    panel = np.zeros((320, 700, 3), np.uint8)
    baris = [
        ("-- HSV --", (0, 255, 255)),
        (f"H1 min : {h1_min}", (255, 255, 255)),
        (f"H1 max : {h1_max}", (255, 255, 255)),
        (f"H2 min : {h2_min}", (255, 255, 255)),
        (f"H2 max : {h2_max}", (255, 255, 255)),
        (f"S min  : {s_min}", (255, 255, 255)),
        (f"V min  : {v_min}", (255, 255, 255)),
        ("-- KAMERA (dikunci) --", (0, 255, 255)),
        (f"Exposure : {exposure}", (255, 255, 255)),
        (f"Gain     : {gain}", (255, 255, 255)),
        (f"WB       : {wb} K", (255, 255, 255)),
        (f"Focus    : {focus}", (255, 255, 255)),
    ]
    for i, (t, warna) in enumerate(baris):
        cv2.putText(panel, t, (10, 22 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, warna, 1, cv2.LINE_AA)
    cv2.imshow(win, panel)

    cv2.imshow("Kamera", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Hasil", hasil)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
        print("\n# ---- Salin ke robot.py ----")
        print(f"LOWER_RED1 = np.array([{h1_min}, {s_min}, {v_min}])")
        print(f"UPPER_RED1 = np.array([{h1_max}, 255, 255])")
        print(f"LOWER_RED2 = np.array([{h2_min}, {s_min}, {v_min}])")
        print(f"UPPER_RED2 = np.array([{h2_max}, 255, 255])")
        print("\ndef lock_camera(cap):")
        print("    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)")
        print(f"    cap.set(cv2.CAP_PROP_EXPOSURE, {exposure})")
        print(f"    cap.set(cv2.CAP_PROP_GAIN, {gain})")
        print("    cap.set(cv2.CAP_PROP_AUTO_WB, 0)")
        print(f"    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, {wb})")
        print("    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)")
        print(f"    cap.set(cv2.CAP_PROP_FOCUS, {focus})")
        print("# ---------------------------\n")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()