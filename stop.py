import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import WebcamModule as wM
import MotorModule as mM
import time

#######################################
# Initialize Parameters
steeringSen = 0.70
maxThrottle = 0.31
motor = mM.Motor(2, 3, 4, 17, 22, 27)

# Initialize Lane Detection Model
interpreter = tflite.Interpreter(model_path='/home/pujan/Desktop/Self/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize YOLO for Stop Sign Detection
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Counter for stop sign detection frequency
detection_counter = 0
DETECTION_FREQUENCY = 10  # Only run stop sign detection every 10 frames

#######################################
def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def detect_stop_sign(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    stop_sign_detected = False

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 11:  # Stop Sign class ID
                stop_sign_detected = True
                if frame is not None:
                    center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype('int')
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Stop Sign ({int(confidence * 100)}%)",
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return stop_sign_detected, frame

#######################################
try:
    stop_frame_count = 0
    STOP_FRAMES_THRESHOLD = 1  # Number of consecutive frames needed to confirm stop sign
    last_stop_time = 0  # Track the last time we stopped
    MIN_TIME_BETWEEN_STOPS = 5  # Minimum seconds between stops
    currently_stopping = False  # Flag to track if we're in the process of stopping

    while True:
        start_time = time.time()

        # Get frame from WebcamModule (original size for lane detection)
        img = wM.getImg(False, size=[240, 120])

        # Process image for lane detection
        img = np.asarray(img)
        processed_img = preProcess(img)
        processed_img = np.expand_dims(processed_img, axis=0).astype(np.float32)

        # Lane detection inference
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        steering = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Only run stop sign detection every DETECTION_FREQUENCY frames
        if detection_counter % DETECTION_FREQUENCY == 0:
            detection_frame = wM.getImg(False, size=[320, 240])
            new_stop_detected, annotated_frame = detect_stop_sign(detection_frame)

            # Check if enough time has passed since last stop
            time_since_last_stop = time.time() - last_stop_time

            if new_stop_detected and time_since_last_stop >= MIN_TIME_BETWEEN_STOPS:
                stop_frame_count += 1
                if stop_frame_count >= STOP_FRAMES_THRESHOLD and not currently_stopping:
                    currently_stopping = True
                    print("Stop sign detected! Stopping...")
                    motor.move(0, 0)  # Stop the vehicle
                    time.sleep(2)  # Stop for 2 seconds
                    last_stop_time = time.time()  # Update last stop time
                    currently_stopping = False
                    stop_frame_count = 0  # Reset the counter
            else:
                stop_frame_count = max(0, stop_frame_count - 1)  # Gradually decrease count

            # Display the annotated frame if available
            if annotated_frame is not None:
                cv2.imshow('Detection View', annotated_frame)

        # Normal lane following when not stopping
        if not currently_stopping:
            motor.move(maxThrottle, -steering * steeringSen)
            print(f"Steering: {steering * steeringSen}")

        detection_counter += 1

        # Calculate and print FPS
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped manually.")
finally:
    cv2.destroyAllWindows()
    motor.stop()