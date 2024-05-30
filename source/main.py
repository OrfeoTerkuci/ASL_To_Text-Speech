"""
This is the main file of the project.
Run for the whole project.
"""

import os
from image_rendering.image_processing import ImageProcessor
from gui.gui import App
from image_rendering.image_recognition import WebcamHandtracking, DataCapture

from models.cnn.cnn import (
    BATCH_SIZE,
    CNN,
    CnnLandMarks,
    EPOCHS,
    LEARNING_RATE,
    MIN_DELTA,
    NUM_CLASSES,
    PATIENCE,
)


def test_cnn(id: int, landmarks: bool = False, reduced: bool = False) -> None:
    if landmarks:
        cnn = CnnLandMarks(
            train_file=f"./dataset/csv_dataset/landmarks/train/landmarks_train{'_6' if reduced else ''}.csv",
            test_file=f"./dataset/csv_dataset/landmarks/test/landmarks_test{'_6' if reduced else ''}.csv",
            val_file=f"./dataset/csv_dataset/landmarks/validation/landmarks_validation{'_6' if reduced else ''}.csv",
        )
    else:
        cnn = CNN(
            train_file=f"./dataset/csv_dataset/pixel/train/train{'_6' if reduced else ''}.csv",
            test_file=f"./dataset/csv_dataset/pixel/test/test{'_6' if reduced else ''}.csv",
            val_file=f"./dataset/csv_dataset/pixel/validation/validation{'_6' if reduced else ''}.csv",
        )
    cnn.load(
        f"./models/cnn/cnn_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}_{MIN_DELTA}_{PATIENCE}_{NUM_CLASSES}_{id}{'_landmarks' if landmarks else ''}.pth"
    )
    # cnn.load("./models/best_models/cnn_1e-07_32_1000_0.004_2_6.pth")
    # Run a quick test on the accuracy of the model on the test set
    cnn.test()
    # Load the model into the webcam and start the hand tracking
    webcam = WebcamHandtracking()
    webcam.load_model(cnn)
    webcam.multi_threading_hand_tracking()


def train_cnn(landmarks: bool = False, reduced: bool = False) -> None:
    if landmarks:
        cnn = CnnLandMarks(
            train_file=f"./dataset/csv_dataset/landmarks/train/landmarks_train{'_6' if reduced else ''}.csv",
            test_file=f"./dataset/csv_dataset/landmarks/test/landmarks_test{'_6' if reduced else ''}.csv",
            val_file=f"./dataset/csv_dataset/landmarks/validation/landmarks_validation{'_6' if reduced else ''}.csv",
        )
    else:
        cnn = CNN(
            train_file=f"./dataset/csv_dataset/pixel/train/train{'_6' if reduced else ''}.csv",
            test_file=f"./dataset/csv_dataset/pixel/test/test{'_6' if reduced else ''}.csv",
            val_file=f"./dataset/csv_dataset/pixel/validation/validation{'_6' if reduced else ''}.csv",
        )
    cnn.train_model()


def benchmark_cnn():
    models = []
    for subdir, _, files in os.walk("./models/best_models/"):
        for file in files:
            if file.endswith(".pth"):
                params = file.split("_")
                if len(params) > 1:
                    models.append((subdir, file))
    for subdir, file in models:
        landmarks = "landmarks" in file
        reduced = file.split("_")[-2] == "6" or file.endswith("6.pth")
        if not reduced:
            continue
        print(f"Model: {file}")
        if landmarks:
            cnn = CnnLandMarks(
                train_file=f"./dataset/csv_dataset/landmarks/train/landmarks_train{'_6' if reduced else ''}.csv",
                test_file=f"./dataset/csv_dataset/landmarks/test/landmarks_test{'_6' if reduced else ''}.csv",
                val_file=f"./dataset/csv_dataset/landmarks/validation/landmarks_validation{'_6' if reduced else ''}.csv",
                train=False,
            )
        cnn = CNN(
            train_file=f"./dataset/csv_dataset/pixel/train/train{'_6' if reduced else ''}.csv",
            test_file=f"./dataset/csv_dataset/pixel/test/test{'_6' if reduced else ''}.csv",
            val_file=f"./dataset/csv_dataset/pixel/validation/validation{'_6' if reduced else ''}.csv",
        )
        print(os.path.join(subdir, file))
        try:
            cnn.load(os.path.join(subdir, file))
        except Exception as e:
            print(e)
            continue
        cnn.test()


def process_images_for_landmarks() -> None:
    ip = ImageProcessor()
    ip.process_images_to_csv_landmarks("./dataset/images/")


def process_images_for_landmarks_indi() -> None:
    ip = ImageProcessor()
    ip.process_images_to_csv_individually("./dataset/images/")

def process_images_crop_to_landmarks() -> None:
    ip = ImageProcessor()
    ip.crop_to_landmarks("./dataset/new/", "./dataset/new/")


def resize_to_standard() -> None:
    ip = ImageProcessor()
    ip.resize_images("./dataset/images/")


if __name__ == "__main__":
    # train_cnn(landmarks=False, reduced=True)
    # test_cnn(id=9034, landmarks=False, reduced=True)
    # process_images_for_landmarks_indi()
    # process_images_crop_to_landmarks()
    # webcam = DataCapture()
    # webcam.capture_data()    
    app = App()
    app.mainloop()
