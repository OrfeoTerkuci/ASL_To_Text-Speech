"""
This file contains all the classes and methods for hand tracking
"""

# pylint: disable=E1101
import copy
import os
import string
import threading
import time
from typing import Any

import cv2
import mediapipe as mp  # type: ignore
import numpy as np
import torch

from models.basemodel import BaseModel
from PIL import Image
from utils import (
    adjust_min_max,
    coords_calc,
    detect_key,
    exit_on_close,
    exit_on_key,
    letter_predictions,
)

# STANDARD VARIABLES
FRAME_WIDTH = 1080
FRAME_HEIGHT = 720
OFFSET = 100


class DataCapture:
    """
    Class to capture data from the webcam
    """

    def __init__(self) -> None:
        self.video: cv2.VideoCapture | None = None

    def start_webcam(self) -> None:
        """
        Start the webcam
        """
        if self.video is None:
            self.video = cv2.VideoCapture(0)

    def get_frame(self, size: tuple[int, int] = (1080, 720)) -> Any:
        """
        Get the frame from the webcam
        :param size: tuple (width, height) of the frame.
        :return: Frame
        :raises ValueError: If the VideoCapture is not initialized
        """
        if self.video is not None:
            _, frame = self.video.read()
            frame = cv2.resize(frame, size)
            return frame
        raise ValueError("VideoCapture not initialized")

    def __del__(self) -> None:
        if self.video is not None:
            self.video.release()

    def capture_data(self) -> None:
        """
        Capture data from the webcam
        """

        self.start_webcam()

        # Go through alphabet and save images
        alphabet = string.ascii_lowercase
        exit_bool = False
        for letter in alphabet:
            if letter in ["j", "z"]:
                # Skip j and z because they are not possible with images
                continue
            print(
                f"Commands:\n"
                f"'s' to save image for letter '{letter}'\n"
                f"'n' to move to the next letter\n"
                f"'q' to exit\n"
                f"--------------------------------------\n"
            )
            while True:
                frame = self.get_frame()
                frame = cv2.flip(frame, 1)
                # Show the current letter
                cv2.putText(
                    frame,
                    f"Recording {letter.upper()}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                )
                # Show instructions
                cv2.putText(
                    frame,
                    "'s' to save image",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                )
                cv2.putText(
                    frame,
                    "'n' to move to the next letter",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                )

                cv2.imshow("frame", frame)
                if cv2.waitKey(10) & 0xFF == ord("s"):
                    # If the directory does not exist, create it
                    if not os.path.exists(f"dataset/new/{letter}_dir"):
                        os.makedirs(f"dataset/new/{letter}_dir")
                    dir_size = len(os.listdir(f"dataset/new/{letter}_dir"))
                    # Read the frame again, because old frame might be outdated
                    frame = self.get_frame()
                    frame = cv2.flip(frame, 1)
                    print(f"Saving dataset/new/{letter}_dir/{letter}{dir_size}.png")

                    cv2.imwrite(
                        f"dataset/new/{letter}_dir/{letter}{dir_size}.png", frame
                    )
                    cv2.imshow("frame", frame)

                if cv2.waitKey(10) & 0xFF == ord("n"):
                    break

                if exit_on_key() or exit_on_close():
                    exit_bool = True
                    break
            if exit_bool:
                break
        cv2.destroyAllWindows()
        print("Data capture completed")
        del self


class MpSettings:  # pylint: disable=R0903
    """
    Class to set the settings for the mediapipe
    """

    def __init__(self, static: bool = False) -> None:
        self.mp_hands = mp.solutions.hands  # type: ignore
        self.hands = self.mp_hands.Hands(
            static_image_mode=static,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils  # type: ignore

    def __del__(self) -> None:
        self.hands.close()


class WebcamHandtracking:  # pylint: disable=R0902
    """
    Class to track the hand using the webcam
    """

    def __init__(self) -> None:
        """
        Initialize the webcam
        """
        self.model: BaseModel = BaseModel("", "", "", False)
        self.video: cv2.VideoCapture | None = None
        self.mp_settings = MpSettings()
        self.p_time: float = 0.0
        self.c_time: float = 0.0
        self.results = None
        self.lm_list: list[list[float]] = []
        self.extreme_points: tuple[int, int, int, int] = (0, 0, 0, 0)
        self.cx_cy: list[tuple[int, int]] = []
        self.last_analyze_time = time.time()
        self.prediction: list[tuple[int, str, float]] = [
            (1, "", 0.0),
            (2, "", 0.0),
            (3, "", 0.0),
        ]

    def load_model(self, model: BaseModel) -> None:
        """
        Load the model
        :param model_type: The type of the model
        """
        self.model = model

    def __del__(self) -> None:
        """
        Release the webcam
        """
        if self.video is not None:
            self.video.release()

    def start_webcam(self) -> None:
        """
        Start the webcam
        """
        if self.video is None:
            self.video = cv2.VideoCapture(0)

    def get_frame(self, size: tuple[int, int] = (1080, 720)) -> Any:
        """
        Get the frame from the webcam
        :param size: tuple (width, height) of the frame.
        :return: Frame
        :raises ValueError: If the VideoCapture is not initialized
        """
        if self.video is not None:
            _, frame = self.video.read()
            frame = cv2.resize(frame, size)
            return frame
        raise ValueError("VideoCapture not initialized")

    def calculate_fps(self) -> float:
        """
        Calculate the frame per second
        :return: The fps
        """
        self.c_time = time.time()
        fps = 1 / (self.c_time - self.p_time)
        self.p_time = self.c_time
        return fps

    def find_hands(self, frame: Any) -> Any:
        """
        Find the hands.
        :param frame: Frame
        :param draw: bool
        :return: frame
        """
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Set the frame to the hand tracking model
        self.results = self.mp_settings.hands.process(frame_rgb)
        return frame

    def find_position(  # pylint: disable=R0914, R0913
        self,
        frame: Any,
        threads: list[threading.Thread],
        hand_no: int = 0,
    ) -> None:
        """
        Find the position of the hand
        :param frame: frame
        :param hand_no: int
        :param draw: bool
        :param threads: list[threading.Thread]
        """
        lm_list: list[list[float]] = []
        # If the hand is detected
        if self.results and self.results.multi_hand_landmarks:
            # Get the hand
            my_hand = self.results.multi_hand_landmarks[hand_no]
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            self.cx_cy = []
            for hand_id, lm in enumerate(my_hand.landmark):
                x_max, x_min, y_max, y_min, cx, cy = coords_calc(
                    hand_id, frame, lm, lm_list, x_max, x_min, y_max, y_min
                )
                self.cx_cy.append((cx, cy))

            x_max, x_min, y_max, y_min = adjust_min_max(
                OFFSET, frame, x_max, x_min, y_max, y_min
            )

            # Get the minimum and maximum coordinates of the box
            self.extreme_points = (x_max, x_min, y_max, y_min)

            # Only predict once per second
            if time.time() - self.last_analyze_time > 1:
                the3 = threading.Thread(
                    target=self.analyze_and_predict, args=(frame, self.model, threads)
                )
                the3.start()
                threads.append(the3)
                self.last_analyze_time = time.time()  # reset the timer

        self.lm_list = lm_list

    def save_position(
        self,
        frame: Any,
        hand_no: int = 0,
        image_no: int = 0,
        save_dir: str = "img_input",
    ) -> None:
        """
        Find the position of the hand and save it at the specified directory.
        Default directory is img_input
        :param frame: The frame to save
        :param hand_no: The index of the hand
        :param image_no: The index of the image
        :param save_dir: The directory to save the image
        """
        lm_list: list[list[float]] = []
        offset = 50
        original_frame = copy.deepcopy(frame)
        # If the hand is detected
        if self.results and self.results.multi_hand_landmarks:
            # Get the hand
            my_hand = self.results.multi_hand_landmarks[hand_no]
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0
            for hand_id, lm in enumerate(my_hand.landmark):
                x_max, x_min, y_max, y_min, _, _ = coords_calc(
                    hand_id, frame, lm, lm_list, x_max, x_min, y_max, y_min
                )

            x_max, x_min, y_max, y_min = adjust_min_max(
                offset, original_frame, x_max, x_min, y_max, y_min
            )

            print(
                f"Saving image hand_img{image_no}.png with shape "
                f"{original_frame[y_min:y_max, x_min:x_max].shape}"
            )
            cv2.imwrite(
                f"{save_dir}/hand_img{image_no}.png",
                original_frame[y_min:y_max, x_min:x_max],
            )
            image_no += 1
        self.lm_list = lm_list

    def analyze_and_predict(
        self, frame: Any, model: BaseModel, threads: list[threading.Thread]
    ) -> None:
        """
        Analyze and predict the hand position

        :param frame: The frame to analyze.
        :param model: The model to use for prediction.
        :param threads: A list of threading.Thread objects.
        """
        analysis_frame = frame
        # Only analyze if a hand is detected
        if self.extreme_points:
            # Convert the frame to grayscale and crop it to the hand
            analysis_frame = analysis_frame[
                self.extreme_points[3] : self.extreme_points[2],
                self.extreme_points[1] : self.extreme_points[0],
            ]
            pixel_data = np.array(analysis_frame)
            # Run the prediction on the model
            predict_thread = threading.Thread(
                target=self.run_prediction,
                args=(model, pixel_data),
            )
            predict_thread.start()
            threads.append(predict_thread)

    def run_prediction(self, model: BaseModel, pixel_data: np.ndarray) -> None:
        """
        Run the prediction on the model.
        :param model: The model to use for prediction
        :param pixel_data: The pixel data
        """

        # Get the prediction from the model
        if (prediction := model.predict(pixel_data)) is None:
            return

        if not prediction.any():
            return

        # Get the top 3 predictions and their indices
        top3_values, top3_indices = torch.topk(torch.from_numpy(prediction), 3)
        # Convert indices to class labels
        top3_predictions = [letter_predictions[i] for i in top3_indices]

        # Add the top 3 predictions to the list and print them
        self.prediction = []
        for i, (value, label) in enumerate(zip(top3_values, top3_predictions)):
            self.prediction.append(
                (i + 1, label, float(value))
            )  # Convert value to float
            print(f"{i + 1}: {label}\tConfidence: {value:.2f}")

    def get_lm_list(self) -> list[list[float]]:
        """
        :return:
        """
        return self.lm_list

    def get_hand_count(self) -> int:
        """
        Get the number of hands
        :return: int
        """
        if self.results and self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0

    def multi_threading_hand_tracking(self) -> None:
        """
        Track the hand using multi-threading
        :param save_dir: The directory to save the images
        :param save_images: bool
        """
        self.start_webcam()
        frame_no = 0
        track = True  # placeholder for now
        while True:
            threads: list[threading.Thread] = []
            frame = self.get_frame()
            frame = cv2.flip(frame, 1)

            if track:
                frame = self.track(frame, frame_no, threads)
                frame_no += 1

            cv2.imshow("frame", frame)

            for thr in threads:
                thr.join()

            if exit_on_key():
                break
            if exit_on_close():
                break

        cv2.destroyAllWindows()

    def track(  # pylint: disable=R0913, R0914
        self,
        frame: Any,
        frame_no: int,
        threads: list[threading.Thread],
    ) -> Any:
        """
        Track the hand and save the position
        :param colors: The colors to use
        :param frame: The frame to track
        :param frame_no: The frame number
        :param original_frame: The original frame (before any changes)
        :param save_images: Whether to save the images
        :param threads: The threads
        :param save_dir: The directory to save the images
        :return: The frame
        """
        frame = self.find_hands(frame)
        count = self.get_hand_count()
        for i in range(count):
            # # Mark the position of the hand
            thr2 = threading.Thread(target=self.find_position, args=(frame, threads, i))
            thr2.start()
            threads.append(thr2)
            frame_no += 1
        self.draw_frame(frame)
        return frame

    def draw_frame(self, frame: Any) -> None:
        """
        Draw the frame
        :param frame: The frame to draw
        """
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
        fps = self.calculate_fps()

        # Draw the landmarks on the frame
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_settings.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_settings.mp_hands.HAND_CONNECTIONS,
                )
            for cx, cy in self.cx_cy:
                cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
            x_max, x_min, y_max, y_min = self.extreme_points
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        count = self.get_hand_count()
        # Display the position of the hand on the frame
        if lm_list := self.get_lm_list():
            for i in range(count):
                # Cycle through the colors and put the text on the frame
                color = colors[i % len(colors)]
                cv2.putText(
                    frame,
                    f"Hand {i + 1}: ({lm_list[i][1]}, {lm_list[i][2]})",
                    (10, 100 + i * 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    color,
                    2,
                )

            if count:
                # Display the prediction on the frame
                for i, key, value in self.prediction:
                    text = f"{i}: {key} {(value * 100):.2f}%"

                    cv2.putText(
                        frame,
                        text,
                        (10, 100 + (i + count) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        1,
                    )

        cv2.putText(
            frame,
            str(int(fps)),
            (10, 70),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 255, 255),
            3,
        )
