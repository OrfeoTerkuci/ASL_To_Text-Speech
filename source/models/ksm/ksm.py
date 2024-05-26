"""
Sequential (Deep Neural Network) model made using Keras (Deep Learning Library)
"""

# pylint: disable=E1101

from typing import Any

import cv2
import keras  # type: ignore
import numpy as np
import pandas as pd
from keras import Input
from keras.callbacks import EarlyStopping, LearningRateScheduler  # type: ignore

# import seaborn as sns  # type: ignore
from keras.layers import (  # type: ignore
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPool2D,
)
from keras.models import load_model, Sequential  # type: ignore
from PIL import Image

# from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
# from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import LabelBinarizer  # type: ignore

# pylint: disable=E0401,E0611
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

from source.models import BaseModel

DATASET_DIR = "../../dataset"

TRAIN_DATASET = f"{DATASET_DIR}/csv_dataset/train/train.csv"
TEST_DATASET = f"{DATASET_DIR}/csv_dataset/test/test.csv"
VALIDATION_DATASET = f"{DATASET_DIR}/csv_dataset/validation/validation.csv"

EXTERNAL_DATASET_DIR = f"{DATASET_DIR}/external_dataset"

EPOCHS = 20
BATCH_SIZE = 256
OUTPUT_SIZE = 24
LEARNING_RATE = 0.001


class KSM(BaseModel):
    """
    Sequential Model using Keras (Deep Learning Library)
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "ksm_model"
        self.model: keras.Sequential = Sequential(name=self.name)
        self.init_layers()

        self.train_data: np.ndarray[np.uint8, Any] = np.array([])
        self.test_data: np.ndarray[np.uint8, Any] = np.array([])
        self.validation_data: np.ndarray[np.uint8, Any] = np.array([])

        self.train_labels: np.ndarray[np.uint8, Any] = np.array([])
        self.test_labels: np.ndarray[np.uint8, Any] = np.array([])
        self.valid_labels: np.ndarray[np.uint8, Any] = np.array([])

        self.history = None

        self.datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False,
        )

    def init_layers(self) -> None:
        """
        Initialize the layers of the model
        """
        # Input layer
        self.model.add(Input(shape=(28, 28, 1)))
        # First layer
        self.model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        # Downsample the data
        self.model.add(MaxPool2D((2, 2), strides=2, padding="same"))

        # Second layer
        self.model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2), strides=2, padding="same"))

        # Third layer
        self.model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D((2, 2), strides=2, padding="same"))

        # Flatten the input
        self.model.add(Flatten())

        # Three dense layers of 512, 1024, 512 and 24 output
        self.model.add(Dense(units=512, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=1024, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=512, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=OUTPUT_SIZE, activation="softmax"))

    def load_data(self) -> None:
        """
        Load the data from the dataset
        """
        # Load the data
        train_df = pd.read_csv(TRAIN_DATASET)
        test_df = pd.read_csv(TEST_DATASET)
        valid_df = pd.read_csv(VALIDATION_DATASET)

        label_binarizer = LabelBinarizer()

        y_train: np.ndarray[np.uint8, Any] = label_binarizer.fit_transform(
            train_df["label"]
        )
        y_test: np.ndarray[np.uint8, Any] = label_binarizer.fit_transform(
            test_df["label"]
        )
        y_valid: np.ndarray[np.uint8, Any] = label_binarizer.fit_transform(
            valid_df["label"]
        )

        del train_df["label"]
        del test_df["label"]
        del valid_df["label"]

        x_train = train_df.to_numpy()
        x_test = test_df.to_numpy()
        x_valid = valid_df.to_numpy()

        x_train = x_train / 255
        x_test = x_test / 255
        x_valid = x_valid / 255

        self.train_data = x_train.reshape(-1, 28, 28, 1)
        self.test_data = x_test.reshape(-1, 28, 28, 1)
        self.validation_data = x_valid.reshape(-1, 28, 28, 1)

        self.train_labels = y_train
        self.test_labels = y_test
        self.valid_labels = y_valid

        # Fit the data with random transformations
        self.datagen.fit(self.train_data)

    @staticmethod
    def lr_decay(epoch: int) -> float:
        """
        Learning rate decay
        :param epoch: The current epoch
        :return: The new learning rate
        """
        drop = 0.5
        epochs_drop = 10.0
        return float(LEARNING_RATE * (drop ** ((1 + epoch) // epochs_drop)))

    def train(self, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> None:
        """
        Compile the model
        :param epochs: The number of epochs
        :param batch_size: The batch size
        """
        print("Compiling the model")

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )

        lr_scheduler = LearningRateScheduler(self.lr_decay)

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["mse", "accuracy"],
        )
        # self.model.summary()

        self.history = self.model.fit(
            self.datagen.flow(
                self.train_data, self.train_labels, batch_size=batch_size
            ),
            epochs=epochs,
            validation_data=(self.validation_data, self.valid_labels),
            callbacks=[early_stopping, lr_scheduler],
            validation_batch_size=batch_size,
        )

        self.model.save(f"{self.name}.keras")

    def load_model(self, model_name: str) -> None:
        """
        Load the model from the file
        :param model_name: The name of the model
        """
        self.model = load_model(model_name, compile=False)

    def print(self) -> None:
        """
        Print the model summary
        """
        print(self.model.summary())

    def predict(self, img: Image.Image) -> Any:
        """
        Predict the label of the pixel data
        :param img: The image to predict
        :return: The label
        """

        # Get the pixel data into a numpy array (28x28x1)
        pixel_data = np.array(img)
        # If it has 3 channels, convert it to 1 channel
        if img.mode == "RGB":
            pixel_data = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2GRAY)

        pixel_data = cv2.resize(pixel_data, (28, 28))
        pixel_data = pixel_data.reshape(-1, 28, 28, 1)

        return self.model.predict(pixel_data, verbose=0)


def get_best_model() -> None:
    """
    Get the best model from the log file
    """
    log_loc = "seq_model_log.csv"
    df = pd.read_csv(log_loc)
    print(df[df["Score"] == df["Score"].max()])


if __name__ == "__main__":
    model = KSM()
    model.load_data()
    model.train()
    # model.load("keras_sequential_model.keras")
    # log_train()
    # get_best_model()
