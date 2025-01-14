from picamera2 import Picamera2, Preview
import cv2

# Initialize the Picamera2 instance
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)  # Set the resolution
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

def getImg(display=False, size=[480, 240]):
    img = picam2.capture_array()
    img = cv2.resize(img, (size[0], size[1]))
    if display:
        cv2.imshow('IMG', img)
    return img

if __name__ == '__main__':
    while True:
        img = getImg(True)
        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
