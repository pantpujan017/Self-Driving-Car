"""
- This module handles data collection for multiple folders
- Each folder can contain multiple images with a single log file
- Parent folder should be created manually with the name "DataCollected"
- Images are saved in numbered folders (IMG0, IMG1, etc.)
- Each folder has its own log file tracking all images in that folder
"""
import pandas as pd
import os
import cv2
from datetime import datetime

class DataCollector:
    def __init__(self):
        # Initialize base directory
        self.base_directory = os.path.join(os.getcwd(), 'DataCollected')
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

        # Initialize tracking variables
        self.current_folder = None
        self.current_folder_images = []
        self.current_folder_steering = []
        self.image_count = 0

        # Create first folder
        self.create_new_folder()

    def create_new_folder(self):
        """Create a new folder for storing images"""
        # Save log of current folder if it exists
        if self.current_folder and self.current_folder_images:
            self.save_current_folder_log()

        # Reset tracking variables
        self.current_folder_images = []
        self.current_folder_steering = []
        self.image_count = 0

        # Create new folder
        folder_num = 0
        while os.path.exists(os.path.join(self.base_directory, f'IMG{folder_num}')):
            folder_num += 1

        self.current_folder = os.path.join(self.base_directory, f'IMG{folder_num}')
        os.makedirs(self.current_folder)
        print(f'Created new folder: {self.current_folder}')

    def save_data(self, img, steering):
        """Save an image and its steering data"""
        if self.current_folder is None:
            self.create_new_folder()

        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

        # Save image
        filename = f'Image_{timestamp}.jpg'
        filepath = os.path.join(self.current_folder, filename)
        cv2.imwrite(filepath, img)

        # Track image and steering data
        self.current_folder_images.append(filepath)
        self.current_folder_steering.append(steering)
        self.image_count += 1

        return filepath

    def save_current_folder_log(self):
        """Save log file for current folder"""
        if not self.current_folder_images:
            return

        # Create DataFrame with image paths and steering values
        log_data = {
            'Image': self.current_folder_images,
            'Steering': self.current_folder_steering
        }
        df = pd.DataFrame(log_data)

        # Save log file in current folder
        log_path = os.path.join(self.current_folder, 'log.csv')
        df.to_csv(log_path, index=False)
        print(f'Log saved: {log_path}')
        print(f'Total images in folder: {self.image_count}')

# For testing when run independently
if __name__ == '__main__':
    collector = DataCollector()
    cap = cv2.VideoCapture(1)

    try:
        # Test with multiple folders
        for folder in range(2):  # Create 2 folders
            if folder > 0:
                collector.create_new_folder()

            # Save 5 images in each folder
            for x in range(5):
                ret, img = cap.read()
                if ret:
                    filepath = collector.save_data(img, 0.5)
                    print(f'Saved image: {filepath}')
                    cv2.imshow("Image", img)
                    cv2.waitKey(1)

            collector.save_current_folder_log()

    finally:
        cap.release()
        cv2.destroyAllWindows()