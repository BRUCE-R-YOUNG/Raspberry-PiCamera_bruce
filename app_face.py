import time
import board
import busio
import cv2
import numpy as np
from adafruit_pca9685 import PCA9685
from adafruit_motor.servo import Servo

# Create the I2C bus interface.
i2c = busio.I2C(board.SCL, board.SDA)

# Create a simple PCA9685 class instance.
pca = PCA9685(i2c)
pca.frequency = 50  # Frequency for servos (50 Hz)

# Create servo instances on channels 0 and 1 (for pan and tilt).
pan_servo = Servo(pca.channels[0])
tilt_servo = Servo(pca.channels[1])

# Set up initial positions.
pan_servo.angle = 90  # Pan center position
tilt_servo.angle = 90  # Tilt center position

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize camera
cap = cv2.VideoCapture(1)  # Use index 1 to access /dev/video1

# Frame width and height
frame_width = 640
frame_height = 480

cap.set(3, frame_width)
cap.set(4, frame_height)

# Servo angles
pan_angle = 90
tilt_angle = 90

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale (required for Haar Cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # If any faces are found, proceed
        if len(faces) > 0:
            # Get the coordinates of the largest detected face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            center_x = x + w // 2
            center_y = y + h // 2

            # Move pan servo to follow the face horizontally
            if center_x < frame_width / 3:
                pan_angle += 2
            elif center_x > 2 * frame_width / 3:
                pan_angle -= 2

            # Move tilt servo to follow the face vertically
            if center_y < frame_height / 3:
                tilt_angle += 2
            elif center_y > 2 * frame_height / 3:
                tilt_angle -= 2

            # Constrain the angles between 0 and 180
            pan_angle = max(0, min(180, pan_angle))
            tilt_angle = max(0, min(180, tilt_angle))

            # Set the servo angles
            pan_servo.angle = pan_angle
            tilt_servo.angle = tilt_angle

        # Show the resulting frame with a rectangle around the detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Face Tracking', frame)

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
