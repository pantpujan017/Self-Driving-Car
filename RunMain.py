import cv2
import numpy as np
import tflite_runtime.interpreter as tflite  # Use the TFLite interpreter
import WebcamModule as wM
import MotorModule as mM

#######################################
steeringSen = 0.70  # Steering Sensitivity
maxThrottle = 0.31  # Forward Speed %
motor = mM.Motor(2, 3, 4, 17, 22, 27)  # Pin Numbers

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='/home/pujan/Desktop/Self/model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
#######################################

def preProcess(img):
    img = img[54:120, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

while True:
    img = wM.getImg(True, size=[240, 120])
    img = np.asarray(img)
    img = preProcess(img)
    img = np.expand_dims(img, axis=0).astype(np.float32)  # Ensure the input matches the model's requirements

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    steering = interpreter.get_tensor(output_details[0]['index'])[0][0]

    print(steering * steeringSen)
    motor.move(maxThrottle, -steering * steeringSen)
    cv2.waitKey(1)