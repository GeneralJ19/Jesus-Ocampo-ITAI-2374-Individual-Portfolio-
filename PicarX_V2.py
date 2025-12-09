import cv2
import numpy as np
import time
from picarx import PiCarX

# Initialize PiCarX
picar = PiCarX()
picar.set_dir_servo_angle(90)  # Center camera servo
picar.forward(0)

# HSV color range for ball detection (e.g., orange/red)
lower_hsv = np.array([5, 150, 150])
upper_hsv = np.array([15, 255, 255])

# PID-like proportional control parameters for servo
servo_center = 90
servo_range = 40
frame_center_x = 160  # assuming 320x240 frame

# Minimum ultrasonic distance to keep moving forward (cm)
safe_distance = 25

def find_ball(frame):
    """Detect ball in frame and return center x, y and radius."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour assumed as the ball
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius > 10:
            return int(x), int(y), int(radius)
    return None, None, None

try:
    # Start camera streaming
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found!")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (320, 240))
        ball_x, ball_y, ball_radius = find_ball(frame)

        # Read ultrasonic distance to avoid obstacles
        distance = picar.get_distance()
        # Safety check
        if distance < safe_distance and distance != -1:
            picar.stop()
            print(f"Obstacle detected at {distance} cm, stopping.")
            time.sleep(0.5)
            continue

        if ball_x is not None:
            # Calculate servo angle based on ball position
            error_x = ball_x - frame_center_x
            angle = servo_center - (error_x / frame_center_x) * servo_range
            angle = max(40, min(140, angle))  # Clamp angle between 40 and 140 degrees
            picar.set_dir_servo_angle(angle)

            # Simple forward movement logic
            if ball_radius < 50:
                picar.forward(30)  # Move forward slowly
                print(f"Following ball at radius {ball_radius}, servo angle {angle:.1f}")
            else:
                picar.stop()
                print("Ball is close, stopped.")
        else:
            picar.stop()
            print("No ball detected, stopped.")

        # Optional: Uncomment to show camera feed for debugging
        # cv2.circle(frame, (ball_x, ball_y), ball_radius, (0, 255, 0), 2)
        # cv2.imshow('Ball Tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Program stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    picar.stop()
