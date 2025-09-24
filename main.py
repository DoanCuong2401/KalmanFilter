import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# --- Tạo Kalman Filter ---
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
kalman.transitionMatrix  = np.array([[1,0,1,0],
                                     [0,1,0,1],
                                     [0,0,1,0],
                                     [0,0,0,1]], np.float32)
kalman.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.03

# --- Mở camera ---
cap = cv2.VideoCapture(0)

# Ngưỡng màu xanh trong HSV
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# --- Khởi tạo figure để plot ---
plt.ion()
fig, ax = plt.subplots()
distances = []
line1, = ax.plot(distances, label="Diff (px)")
ax.set_ylim(0, 50)  # tùy chỉnh trục Y
ax.set_xlim(0, 100) # hiển thị 100 điểm gần nhất
ax.legend()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Chuyển sang HSV và tạo mask màu xanh ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- Tìm contour lớn nhất ---
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        if radius > 5:
            center = np.array([[np.float32(x)], [np.float32(y)]])
            cv2.circle(frame, (int(x), int(y)), 10, (0,0,255), -1)  # đỏ

    # --- Kalman Filter ---
    prediction = kalman.predict()
    pred_pt = (int(prediction[0]), int(prediction[1]))
    cv2.circle(frame, pred_pt, 10, (0,255,0), 2)  # xanh lá

    if center is not None:
        kalman.correct(center)
        raw_pt = (int(center[0]), int(center[1]))
        dist = math.hypot(raw_pt[0] - pred_pt[0], raw_pt[1] - pred_pt[1])
        cv2.line(frame, raw_pt, pred_pt, (255,255,0), 1)
        cv2.putText(frame, f"Diff: {dist:.2f}px", (raw_pt[0]+10, raw_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # --- Cập nhật dữ liệu plot ---
        distances.append(dist)
        if len(distances) > 100:
            distances = distances[-100:]  # chỉ giữ 100 điểm gần nhất

        frame_count += 1
        if frame_count % 5 == 0:  # update đồ thị mỗi 5 frame
            line1.set_ydata(distances)
            line1.set_xdata(range(len(distances)))
            ax.set_xlim(0, len(distances))
            ax.set_ylim(0, max(50, max(distances)+10))
            fig.canvas.draw()
            fig.canvas.flush_events()

    cv2.imshow("Blue Pen Tracking (Red=raw, Green=Kalman)", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
