import time
import board
import busio
import cv2
from adafruit_pca9685 import PCA9685
from adafruit_motor.servo import Servo
from ultralytics import YOLO

# Create the I2C bus interface.
i2c = busio.I2C(board.SCL, board.SDA)

# Create a simple PCA9685 class instance.
pca = PCA9685(i2c)
pca.frequency = 50  # Frequency for servos (50 Hz)

# Create servo instances on channels 0 and 1 (for pan and tilt).
pan_servo = Servo(pca.channels[0])
tilt_servo = Servo(pca.channels[1])

# Set up initial positions.
default_pan_angle = 90
default_tilt_angle = 90
pan_servo.angle = default_pan_angle  # Pan center position
tilt_servo.angle = default_tilt_angle  # Tilt center position

# Initialize camera
cap = cv2.VideoCapture(1)  # Use index 1 to access /dev/video1

# Frame width and height (reduce resolution to improve performance)
frame_width = 320
frame_height = 240

cap.set(3, frame_width)
cap.set(4, frame_height)

# Servo angles
pan_angle = default_pan_angle
tilt_angle = default_tilt_angle

# Load the YOLO model (YOLOv8)
model = YOLO('yolov8n.pt')  # Use the nano model for better performance on Raspberry Pi

# Frame skipping counter to improve performance
frame_count = 0
frame_skip = 5  # Skip 4 out of every 5 frames

# Set the number of frames after which to reset to default if no detection
no_detection_frame_threshold = 15
no_detection_counter = 0

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame (mirror effect) for better user interaction
        frame = cv2.flip(frame, 1)

        # Only run the detection every frame_skip frames to improve performance
        if frame_count % frame_skip == 0:
            results = model(frame)

            # If objects are detected, proceed
            detected = False
            for result in results:
                for box in result.boxes:
                    # If the detected object is a person (class index 0 is 'person' in YOLO)
                    if int(box.cls) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Move pan servo to follow the person horizontally
                        if center_x < frame_width / 3:
                            pan_angle += 2
                        elif center_x > 2 * frame_width / 3:
                            pan_angle -= 2

                        # Move tilt servo to follow the person vertically
                        if center_y < frame_height / 3:
                            tilt_angle += 2
                        elif center_y > 2 * frame_height / 3:
                            tilt_angle -= 2

                        # Constrain the angles between 0 and 180
                        pan_angle = max(0, min(180, pan_angle))
                        tilt_angle = max(0, min(180, tilt_angle))

                        # Reset the no-detection counter
                        no_detection_counter = 0
                        detected = True

            # If nothing is detected, increase the counter
            if not detected:
                no_detection_counter += 1

        # Set the servo angles (update every frame for smooth movement)
        if no_detection_counter > no_detection_frame_threshold:
            # Reset to default positions if nothing is detected for a while
            pan_angle = default_pan_angle
            tilt_angle = default_tilt_angle

        pan_servo.angle = pan_angle
        tilt_servo.angle = tilt_angle

        # Draw the rectangle around the detected person
        if 'x1' in locals():
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Show the resulting frame
        cv2.imshow('YOLO Tracking', frame)

        # Increment frame count
        frame_count += 1

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping program.")

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

# Deinitialize the PCA9685 to stop sending PWM signals.
pca.deinit()
