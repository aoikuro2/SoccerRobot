import cv2
import numpy as np

cap = cv2.VideoCapture(3, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

win = "Kalibrasi"
cv2.namedWindow(win)

def nothing(x):
    pass

# (nama, nilai awal, maksimum)
sliders = [
    ("H1 min", 0, 179), ("H1 max", 15, 179),     # merah-oranye (awal roda warna)
    ("H2 min", 170, 179), ("H2 max", 179, 179),  # merah (ujung roda warna)
    ("S min", 90, 255),                          # seberapa "pekat" warnanya
    ("V min", 60, 255),                          # seberapa terang
]
for name, val, mx in sliders:
    cv2.createTrackbar(name, win, val, mx, nothing)

print("Tekan P untuk cetak nilai, Q untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    g = lambda n: cv2.getTrackbarPos(n, win)
    s_min, v_min = g("S min"), g("V min")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (g("H1 min"), s_min, v_min), (g("H1 max"), 255, 255))
    m2 = cv2.inRange(hsv, (g("H2 min"), s_min, v_min), (g("H2 max"), 255, 255))
    mask = cv2.bitwise_or(m1, m2)

    hasil = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Kamera", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Hasil", hasil)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("p"):
        print(f"RANGE1: [{g('H1 min')}, {s_min}, {v_min}] - [{g('H1 max')}, 255, 255]")
        print(f"RANGE2: [{g('H2 min')}, {s_min}, {v_min}] - [{g('H2 max')}, 255, 255]")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()