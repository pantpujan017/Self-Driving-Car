import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import WebcamModule as wM
import MotorModule as mM
import time
from picamera2 import Picamera2

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

# Initialize YOLO
net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define vehicle classes
vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Detection parameters
DETECTION_FREQUENCY = 10  # Run detection every 5 frames
FRAMES_THRESHOLD =1  # Number of consecutive frames needed to confirm detection

# Tracking parameters for each type of detection
detection_states = {
    'stop_sign': {
        'frame_count': 0,
        'last_stop_time': 0,
        'currently_stopping': False,
        'min_time_between_stops': 5,
        'stop_duration': 2
    },
    'traffic_light': {
        'frame_count': 0,
        'last_stop_time': 0,
        'currently_stopping': False,
        'min_time_between_stops': 2,
        'stop_duration': {'Red': 3, 'Yellow': 2, 'Green': 0},
        'current_color': 'Unknown',
        'color_confidence': 0
    },
    'vehicle': {
        'frame_count': 0,
        'last_stop_time': 0,
        'currently_stopping': False,
        'min_time_between_stops': 3,
        'stop_duration': 5
    }
}

def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

def detect_light_color(frame, x, y, w, h):
    margin = int(0.15 * w)
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = w + 2 * margin
    h = h + 2 * margin

    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return "Unknown", 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])

    height = roi.shape[0]
    third = height // 3

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    red_top = np.sum(red_mask[0:third, :]) / (third * roi.shape[1])
    yellow_middle = np.sum(yellow_mask[third:2*third, :]) / (third * roi.shape[1])

    threshold = 20
    confidence = 0

    if red_top > threshold:
        confidence = min(100, red_top)
        return "Red", confidence
    elif yellow_middle > threshold:
        confidence = min(100, yellow_middle)
        return "Yellow", confidence
    else:
        bottom_brightness = np.mean(hsv[2*third:, :, 2])
        if bottom_brightness > 100:
            confidence = min(100, bottom_brightness)
            return "Green", confidence
    return "Unknown", 0

def process_detections(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    detected_objects = {
        'stop_sign': {'detected': False, 'confidence': 0},
        'traffic_light': {'detected': False, 'color': 'Unknown', 'confidence': 0},
        'vehicle': {'detected': False, 'confidence': 0, 'type': None}
    }

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype('int')
                x, y = int(center_x - w/2), int(center_y - h/2)

                # Stop Sign Detection (class_id 11)
                if class_id == 11:
                    detected_objects['stop_sign'] = {'detected': True, 'confidence': confidence * 100}
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Stop Sign ({int(confidence * 100)}%)",
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Traffic Light Detection (class_id 9)
                elif class_id == 9:
                    light_color, color_confidence = detect_light_color(frame, x, y, w, h)
                    detected_objects['traffic_light'] = {
                        'detected': True,
                        'color': light_color,
                        'confidence': color_confidence
                    }
                    color = (0, 0, 255) if light_color == "Red" else \
                           (0, 255, 255) if light_color == "Yellow" else \
                           (0, 255, 0) if light_color == "Green" else (255, 255, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"Traffic Light: {light_color} ({int(color_confidence)}%)",
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Vehicle Detection
                elif class_id in vehicle_classes:
                    detected_objects['vehicle'] = {
                        'detected': True,
                        'confidence': confidence * 100,
                        'type': vehicle_classes[class_id]
                    }
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, f"{vehicle_classes[class_id]} ({int(confidence * 100)}%)",
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return detected_objects, frame

def update_detection_state(detection_type, detected, confidence=0, color=None):
    state = detection_states[detection_type]
    current_time = time.time()

    # Reset frame count if color changes for traffic light
    if (detection_type == 'traffic_light' and
        color and
        state['current_color'] != color):
        state['frame_count'] = 0
        state['current_color'] = color
        state['color_confidence'] = confidence

    if detected:
        state['frame_count'] += 1
    else:
        state['frame_count'] = max(0, state['frame_count'] - 1)
        if detection_type == 'traffic_light':
            state['current_color'] = 'Unknown'
            state['color_confidence'] = 0

    # Check if we should stop
    if (state['frame_count'] >= FRAMES_THRESHOLD and
        not state['currently_stopping'] and
        current_time - state['last_stop_time'] >= state['min_time_between_stops']):

        state['currently_stopping'] = True

        # Determine stop duration for traffic lights
        if detection_type == 'traffic_light':
            stop_duration = state['stop_duration'].get(state['current_color'], 0)
            # Only stop for red lights with sufficient confidence
            if state['current_color'] == 'Red' and state['color_confidence'] >= 20:  # Minimum confidence threshold
                print(f"Red Light detected! Confidence: {state['color_confidence']:.1f}% - Stopping for {stop_duration} seconds")
                motor.move(0, 0)
                time.sleep(stop_duration)
                state['last_stop_time'] = time.time()
        else:
            stop_duration = state['stop_duration']
            if stop_duration > 0:
                print(f"{detection_type.replace('_', ' ').title()} detected! " +
                      f"Confidence: {confidence:.1f}% - Stopping for {stop_duration} seconds")
                motor.move(0, 0)
                time.sleep(stop_duration)
                state['last_stop_time'] = time.time()

        state['currently_stopping'] = False
        state['frame_count'] = 0
        return True

    return False

#######################################
try:
    detection_counter = 0

    while True:
        start_time = time.time()

        # Get frame and process for lane detection
        img = wM.getImg(False, size=[240, 120])
        img = np.asarray(img)
        processed_img = preProcess(img)
        processed_img = np.expand_dims(processed_img, axis=0).astype(np.float32)

        # Lane detection inference
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        steering = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Object detection
        if detection_counter % DETECTION_FREQUENCY == 0:
            detection_frame = wM.getImg(False, size=[416, 416])
            detected_objects, annotated_frame = process_detections(detection_frame)

            # Handle detections with priority
            stopping = False

            # Check traffic lights first
            if detected_objects['traffic_light']['detected']:
                stopping = update_detection_state('traffic_light',
                                               True,
                                               detected_objects['traffic_light']['confidence'],
                                               detected_objects['traffic_light']['color'])

            # Then check stop signs if not already stopping
            if not stopping and detected_objects['stop_sign']['detected']:
                stopping = update_detection_state('stop_sign',
                                               True,
                                               detected_objects['stop_sign']['confidence'])

            # Finally check vehicles if not already stopping
            if not stopping and detected_objects['vehicle']['detected']:
                stopping = update_detection_state('vehicle',
                                               True,
                                               detected_objects['vehicle']['confidence'])

            # Display the annotated frame
            if annotated_frame is not None:
                cv2.imshow('Detection View', annotated_frame)

        # Normal lane following when not stopping
        if not any(state['currently_stopping'] for state in detection_states.values()):
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