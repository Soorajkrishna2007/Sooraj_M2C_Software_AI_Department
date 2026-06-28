import cv2
import numpy as np

cap = cv2.VideoCapture(1)

prev_angle = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        gray,
        80,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)

        if area > 5000:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
            data_pts = np.array(contour, dtype=np.float64)
            data_pts = data_pts.reshape(-1, 2)

            mean = np.empty((0))
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(
                data_pts,
                mean
            )

            center = (
                int(mean[0, 0]),
                int(mean[0, 1])
            )

            angle = np.arctan2(
                eigenvectors[0, 1],
                eigenvectors[0, 0]
            )

            angle = np.degrees(angle)

            if angle < 0:
                angle += 180

            angle = 0.8 * prev_angle + 0.2 * angle
            prev_angle = angle

            cv2.drawContours(
                frame,
                [contour],
                -1,
                (0, 255, 0),
                2
            )

            vx, vy, x0, y0 = cv2.fitLine(
                contour,
                cv2.DIST_L2,
                0,
                0.01,
                0.01
            )

            vx = float(vx[0])
            vy = float(vy[0])
            x0 = float(x0[0])
            y0 = float(y0[0])

            length = 220

            p1 = (
                int(center[0] - vx * length),
                int(center[1] - vy * length)
            )

            p2 = (
                int(center[0] + vx * length),
                int(center[1] + vy * length)
            )

            cv2.line(
                frame,
                p1,
                p2,
                (255,0,0),
                3
            )

            angle = np.degrees(np.arctan2(vy, vx))

            if angle < 0:
                angle += 180

            angle = 0.8 * prev_angle + 0.2 * angle
            prev_angle = angle

            x, y, w, h = cv2.boundingRect(contour)

            cv2.putText(
                frame,
                f"Angle: {angle:.1f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Area: {int(area)}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    cv2.imshow("Live Orientation Detection", frame)
    cv2.imshow("Threshold", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()