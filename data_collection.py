import cv2
from time import sleep, time
import DataCollectionModule as dcM
import MotorModule as mM
import KeyboardModule as kbM
import pygame
from picamera2 import Picamera2

def main():
    # Initialize Pi Camera 2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (240, 120)}))
    picam2.start()
    sleep(1)  # Give camera time to warm up

    # Create a named window for display
    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    # Initialize modules
    kbM.init()
    motor = mM.Motor(2, 3, 4, 17, 22, 27)
    data_collector = dcM.DataCollector()

    # Constants
    MAX_THROTTLE = 0.25  # Maximum speed (25% of full speed)
    FRAME_INTERVAL = 0.1  # Capture frame every 100ms

    # State variables
    recording = False
    last_r_key_state = False
    last_f_key_state = False  # For folder switching
    last_capture_time = time()

    print("Controls:")
    print("W: Forward")
    print("S: Backward")
    print("A: Left")
    print("D: Right")
    print("R: Toggle Recording")
    print("F: New Folder")
    print("ESC: Quit")

    try:
        while True:
            # Process pygame events to keep the window responsive
            pygame.event.pump()

            # Capture and display frame
            frame = picam2.capture_array()
            cv2.imshow("Camera Feed", frame)

            # Get keyboard input
            keyVal = kbM.getKeyInput()

            # Debug print for keyboard values
            print(f"Key values - W:{keyVal['W']} S:{keyVal['S']} A:{keyVal['A']} D:{keyVal['D']}", end='\r')

            # Handle recording toggle (R key)
            if keyVal['R'] and not last_r_key_state:
                recording = not recording
                if recording:
                    print('\nRecording Started ...')
                else:
                    print('\nRecording Stopped ...')
                    data_collector.save_current_folder_log()
                sleep(0.1)
            last_r_key_state = keyVal['R']

            # Handle new folder creation (F key)
            if keyVal['F'] and not last_f_key_state:
                if recording:
                    recording = False
                    data_collector.save_current_folder_log()
                data_collector.create_new_folder()
                print('\nCreated new folder for recording...')
                sleep(0.1)
            last_f_key_state = keyVal['F']

            # Calculate steering and throttle from keyboard input
            steering = float(keyVal['D'] - keyVal['A'])  # -1 (left) to 1 (right)
            throttle = float(keyVal['W'] - keyVal['S']) * MAX_THROTTLE  # Forward/Backward with speed limit

            # Move the robot
            motor.move(throttle, -steering)

            # Record data if enabled
            if recording and time() - last_capture_time > FRAME_INTERVAL:
                data_collector.save_data(frame, steering)
                last_capture_time = time()

            # Check for quit condition (ESC key or window close)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

            # Small delay to prevent CPU overload
            sleep(0.01)

    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Cleanup
        print("\nCleaning up...")
        if recording:
            data_collector.save_current_folder_log()
        motor.stop()
        cv2.destroyAllWindows()
        picam2.stop()
        pygame.quit()

if __name__ == '__main__':
    main()