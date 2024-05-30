"""
This module contains the ImageProcessor class.
"""

# pylint: disable=E1101
import os
from typing import Any

import cv2
import mediapipe as mp  # type: ignore
import numpy as np

import pandas as pd

import utils
import csv
from enum import Enum

COLOR_SIZE = 3

PROCESSING_DIR = "preprocessing"
OFFSET = 50

class Letters(Enum):
    """
    This class contains the letters of the alphabet.
    """
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    J = 9
    K = 10
    L = 11
    M = 12
    N = 13
    O = 14
    P = 15
    Q = 16
    R = 17
    S = 18
    T = 19
    U = 20
    V = 21
    W = 22
    X = 23
    Y = 24
    Z = 25

class MpSettings:  # pylint: disable=R0903
    """
    This class contains the settings for the mediapipe module.
    """

    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands  # type: ignore
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore
        self.mp_styles = mp.solutions.drawing_styles  # type: ignore

    def __del__(self) -> None:
        self.hands.close()


class ImageProcessor:
    """
    This class processes the part of the image where the hands are detected.
    It flattens the image and returns it as a 2D array.
    It makes the image grayscale and returns it as a 2D array.
    It makes the image black and white and returns it as a 2D array.
    It
    """

    def __init__(
            self, image: np.ndarray[np.uint8, Any] | None = None
    ) -> None:
        if image is None:
            image = np.zeros((700, 700, 3), np.uint8)
        self.image = image
        self.landmarks: list[list[float]] = []
        self.results = None
        self.mp_settings = MpSettings()

    def flatten_image(self) -> None:
        """
        This method flattens the image and returns it as a 2D array.
        """
        self.grayscale_image()
        if self.image.dtype != np.uint8:
            self.image = cv2.convertScaleAbs(self.image)
        self.image = cv2.equalizeHist(self.image)

    def grayscale_image(self) -> None:
        """
        This method makes the image grayscale and returns it as a 2D array.
        """
        if len(self.image.shape) == COLOR_SIZE and self.image.shape[2] == COLOR_SIZE:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def black_white_image(self) -> None:
        """
        This method makes the image black and white and returns it as a 2D array.
        """
        self.grayscale_image()
        _, self.image = cv2.threshold(self.image, 70, 255, cv2.THRESH_BINARY)

    def resize_image(self, width: int, height: int) -> None:
        """
        This method resizes the image to a specific size.
        :param width: The width to resize the image to.
        :param height: The height to resize the image to.
        """
        self.image = cv2.resize(self.image, (width, height))

    def resize_standard(self) -> None:
        """
        This method resizes the image to a specific size of 28x28.
        """
        self.image = cv2.resize(self.image, (28, 28))

    def crop_image(self, x: int, y: int, width: int, height: int) -> None:
        """
        This method crops the image to a specific size.
        :param x: The x coordinate to start cropping from.
        :param y: The y coordinate to start cropping from.
        :param width: The width to crop the image to.
        :param height: The height to crop the image to.
        """
        self.image = self.image[y: y + height, x: x + width]

    def rotate_image(self, angle: float) -> None:
        """
        This method rotates the image to a specific angle.
        :param angle: The angle to rotate the image to.
        """
        # Get the image's dimensions
        (h, w) = self.image.shape[:2]
        # Compute the center of the image
        center = (w / 2, h / 2)
        # Compute the rotation matrix
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Perform the rotation
        self.image = cv2.warpAffine(self.image, m, (w, h))

    def flip_image(self) -> None:
        """
        This method flips the image horizontally or vertically.
        """
        self.image = cv2.flip(self.image, 1)

    def blur_image(self, kernel_size: int = 5, sigma_x: int = 0) -> None:
        """
        This method blurs the image.
        :param kernel_size: The size of the kernel to use for blurring.
        :param sigma_x: The standard deviation in the x direction.
        """
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), sigma_x)

    def sharpen_image(self, kernel_size: int = 5) -> None:
        """
        This method sharpens the image.
        :param kernel_size: The size of the kernel to use for sharpening.
        """
        self.image = cv2.Laplacian(self.image, cv2.CV_64F, ksize=kernel_size)

    def save_image(self, filename: str) -> None:
        """
        This method saves the image to a specific location.
        :param filename: The location to save the image to.
        """
        cv2.imwrite(filename, self.image)

    def load_image(self, filename: str) -> None:
        """
        This method loads the image from a specific location.
        :param filename: The location to load the image from.
        """
        self.image = cv2.imread(filename)

    def preprocess_image(self, filename: str) -> None:
        """
        This method preprocesses the image.
        :param filename: The location to save the preprocessed image to.
        """
        print("Preprocessing image: " + filename)
        self.resize_standard()
        self.save_image(filename)  # Save the preprocessed image

    def subtract_background(self) -> None:
        """
        This method subtracts the background from the image.
        """
        # Create a background subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        # Apply the background subtractor
        self.image = bg_subtractor.apply(self.image)

    def radial_blur_image(self, amount: int = 30) -> None:
        """
        This method applies a radial blur to the image.
        :param amount: The amount of blur to apply.
        """
        # Create a zero-filled 2D array for the kernel
        kernel = np.zeros((amount, amount), dtype=np.float32)
        # Compute the center of the kernel
        center = (amount - 1) / 2
        # Fill the kernel with weights that decrease as you move away from the center
        for i in range(amount):
            for j in range(amount):
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                kernel[i, j] = np.exp(-(dist ** 2) / (2 * (center ** 2)))
        # Normalize the kernel
        kernel /= np.sum(kernel)
        # Apply the radial blur
        self.image = cv2.filter2D(self.image, -1, kernel)

    def canny_edge_detection(
            self, low_threshold: int = 50, high_threshold: int = 150
    ) -> None:
        """
        This method applies Canny edge detection to the image.
        :param low_threshold: The low threshold for the edge detection.
        :param high_threshold: The high threshold for the edge detection.
        """
        self.image = cv2.Canny(self.image, low_threshold, high_threshold)

    def detect_hands(self) -> None:
        """
        This method detects the hands in the image
        """
        self.results = None
        self.results = self.mp_settings.hands.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        if self.results and not self.results.multi_hand_landmarks:
            print("No hands detected")

    def crop_to_hands(self) -> None:
        """
        This method crops the image to the part where the hands are detected.
        """
        # If the hand is detected
        if self.results and self.results.multi_hand_landmarks:
            for _, my_hand in enumerate(self.results.multi_hand_landmarks):
                x_min, y_min = self.image.shape[1], self.image.shape[0]
                x_max, y_max = 0, 0
                for hand_id, lm in enumerate(my_hand.landmark):
                    x_max, x_min, y_max, y_min, _, _ = utils.coords_calc(
                        hand_id,
                        self.image,
                        lm,
                        self.landmarks,
                        x_max,
                        x_min,
                        y_max,
                        y_min,
                    )
                x_max, x_min, y_max, y_min = utils.adjust_min_max(
                    OFFSET, self.image, x_max, x_min, y_max, y_min
                )
                self.crop_image(x_min, y_min, x_max - x_min, y_max - y_min)

    def draw_to_hands(self) -> None:
        """
        This method draws on the image where the hands are detected.
        """
        # If the hand is detected
        if self.results and self.results.multi_hand_landmarks:
            for _, my_hand in enumerate(self.results.multi_hand_landmarks):
                for _, lm in enumerate(my_hand.landmark):
                    self.mp_settings.mp_draw.draw_landmarks(
                        self.image,
                        my_hand,
                        self.mp_settings.mp_hands.HAND_CONNECTIONS,
                        self.mp_settings.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_settings.mp_styles.get_default_hand_connections_style(),
                    )
                    cx, cy = int(lm.x * self.image.shape[1]), int(
                        lm.y * self.image.shape[0]
                    )
                    cv2.circle(self.image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

    def process_directory(self, input_location: str, output_location: str) -> None:
        """
        This method processes all the images in a directory.
        :param input_location: The location of the input folder.
        :param output_location: The location of the output folder.
        """
        for file in os.listdir(input_location):
            if file.endswith(".png"):
                self.load_image(os.path.join(input_location, file))
                self.preprocess_image(os.path.join(output_location, file))

    def process_landmarks_image(self, input_location: str) -> pd.DataFrame:
        """
        This method processes the landmarks of an image.
        :param input_location: The location of the input image.
        """
        # Read the image using OpenCV
        image = cv2.imread(input_location)
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and get the results
        results = self.mp_settings.hands.process(image_rgb)
        
        # If the hand is detected
        if results.multi_hand_landmarks:
            # Create a list to store the landmarks
            landmarks = []
            if len(results.multi_hand_landmarks) > 1:
                print(f"More than one hand detected for image {input_location}")
                return pd.DataFrame([[]])
            hand_landmarks = results.multi_hand_landmarks[0]
            # Loop through the landmarks
            for landmark in hand_landmarks.landmark:
                # Get the x, y, and z coordinates of the landmark
                x = landmark.x
                y = landmark.y
                # Append the coordinates to the list
                landmarks.append([x, y])
            # Convert the landmarks to a DataFrame
            try:
                landmarks_df = pd.DataFrame([landmarks], columns=[f"landmark_{i}" for i in range(21)])
            except ValueError as e:
                print(landmarks)
                print(len(landmarks))
                raise e
            return landmarks_df
        else:
            # Return an empty DataFrame if no hand is detected
            return pd.DataFrame([[]])

    def process_images_to_csv_landmarks(self, input_location: str) -> None:
        """
        This method processes all the images in a directory and saves the landmarks to a CSV file.
        :param input_location: The location of the input folder.
        """
        with open("landmarks_dataset.csv", "w", newline='', encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            labels = ["label"] + [f"landmark_{i}" for i in range(21)]
            csv_writer.writerow(labels)
            
            for dir_name in os.listdir(input_location):
                dir_path = os.path.join(input_location, dir_name)
                if os.path.isdir(dir_path):  # Ensure it's a directory
                    for file in os.listdir(dir_path):
                        if file.endswith(".png"):
                            image_path = os.path.join(dir_path, file)
                            landmarks_df = self.process_landmarks_image(image_path)
                            if not landmarks_df.empty:
                                # Add label to the dataframe
                                landmarks_df.insert(0, "label", dir_name)
                                # Convert landmarks to a single row with JSON-like format
                                converted_dir = dir_name.replace("_dir", "")
                                letter = Letters[converted_dir.upper()].value
                                row = [letter] + [landmarks_df.iloc[0][f"landmark_{i}"] for i in range(21)]
                                csv_writer.writerow(row)
            
            # randomize the dataset
            df = pd.read_csv("landmarks_dataset.csv")
            df = df.sample(frac=1)
            df.to_csv("landmarks_dataset.csv", index=False)

    def process_images_to_csv_individually(self, input_location: str) -> None:
        """
        This method processes all the images in a directory and saves the landmarks to a CSV file.
        :param input_location: The location of the input folder.
        """
        for dir_name in os.listdir(input_location):
            dir_path = os.path.join(input_location, dir_name)
            if os.path.isdir(dir_path):
                with open(f"{dir_name}.csv", "w", newline='', encoding="utf-8") as f:
                    csv_writer = csv.writer(f)
                    labels = [f"landmark_{i}" for i in range(21)]
                    csv_writer.writerow(labels)
                    for file in os.listdir(dir_path):
                        if file.endswith(".png"):
                            image_path = os.path.join(dir_path, file)
                            landmarks_df = self.process_landmarks_image(image_path)
                            if not landmarks_df.empty:
                                row = [landmarks_df.iloc[0][f"landmark_{i}"] for i in range(21)]
                                csv_writer.writerow(row)


    def crop_to_landmarks(self, input_location: str, output_location: str) -> None:
        """
        This method crops the images to the part where the hands are detected.
        :param input_location: The location of the input folder.
        :param output_location: The location of the output folder.
        """
        for dir_name in os.listdir(input_location):
            dir_path = os.path.join(input_location, dir_name)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".png"):
                        print("Cropping image: " + file)
                        self.load_image(os.path.join(dir_path, file))
                        self.detect_hands()
                        if not self.results:
                            print(f"No hands detected for image {file}")
                            continue
                        if not self.results.multi_hand_landmarks:
                            print(f"No hands detected for image {file}")
                            continue
                        self.crop_to_hands()
                        # self.resize_standard()
                        self.save_image(os.path.join(dir_path, file))


    def resize_images(self, input_location: str) -> None:
        """
        This method resizes the images to a specific size.
        :param input_location: The location of the input folder.
        """
        for dir_name in os.listdir(input_location):
            dir_path = os.path.join(input_location, dir_name)
            if os.path.isdir(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".png"):
                        print("Resizing image: " + file)
                        self.load_image(os.path.join(dir_path, file))
                        self.resize_standard()
                        self.save_image(os.path.join(dir_path, file))



def clear_preprocess_output(output_location: str) -> None:
    """
    This function clears the preprocess_output folder of all images.
    :param output_location: The location of the preprocess_output folder.
    """
    for file in os.listdir(output_location):
        if file.endswith(".png") or file.endswith(".jpg"):
            os.remove(os.path.join(output_location, file))


if __name__ == "__main__":
    processor = ImageProcessor()
    clear_preprocess_output(PROCESSING_DIR)
    processor.process_directory("dataset\\a_dir", PROCESSING_DIR + "\\a_dir")
