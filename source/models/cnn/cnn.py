"""
This module contains the CNN class which is used to create a CNN model.
"""

import os
import random
import sys

# Imports
from typing import Any

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as func
import torch.utils.data as data
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToTensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_rendering.image_processing import MpSettings
from models import basemodel as bm
from utils import adjust_min_max, coords_calc

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-07
EPOCHS = 100000
MIN_DELTA = 0.004
PATIENCE = 4

# Other constants
IN_CHANNELS = 1
INPUT_SIZE = IN_CHANNELS * 28 * 28
INPUT_SIZE_LANDMARKS = 21 * 2
NUM_CLASSES = 6


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Initializes the Trainer class.

        Args:
            model (nn.Module): The CNN model.
            train_loader (DataLoader): The data loader for training data.
            val_loader (DataLoader): The data loader for validation data.
            test_loader (DataLoader): The data loader for test data.
            optimizer (torch.optim.Optimizer): The optimizer for the model.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.id = random.randint(0, 100000)  # Random ID for tensorboard
        self.writer = SummaryWriter(
            comment=f"_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}_{MIN_DELTA}_{PATIENCE}_{NUM_CLASSES}_{self.id}{'_landmarks' if isinstance(self.model, CnnLandMarks) else ''}",
            filename_suffix=f"_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}_{MIN_DELTA}_{PATIENCE}_{NUM_CLASSES}_{self.id}{'_landmarks' if isinstance(self.model, CnnLandMarks) else ''}",
        )
        self.optimizer = optimizer
        self.early_stop_model = bm.EarlyStopping(MIN_DELTA, PATIENCE)

    def early_stop(self, val_loss: float, train_loss: float) -> bool:
        """
        Checks if early stopping criteria is met.

        Args:
            val_loss (float): The validation loss.
            train_loss (float): The training loss.

        Returns:
            bool: True if early stopping criteria is met, False otherwise.
        """
        return self.early_stop_model.early_stop_check(val_loss, train_loss)

    def cuda_available(self) -> bool:
        """
        Checks if cuda is available.

        Returns:
            bool: True if cuda is available, False otherwise.
        """
        return torch.cuda.is_available()

    def train_model(self):
        """
        Trains the CNN model.
        """
        print("Training the model...")

        cuda_available = self.cuda_available()

        if cuda_available:
            self.model.to("cuda")
            print("CUDA available. Training on GPU.")
        else:
            print("CUDA not available. Training on CPU.")

        print("Model: ", self.model, "with id: ", self.id)

        for epoch in range(EPOCHS):
            print("Training...")
            train_loss = 0
            train_correct = 0
            train_size = 0
            for i, batch in enumerate(self.train_loader):
                if cuda_available:
                    batch = (batch[0].to("cuda"), batch[1].to("cuda"))
                # Train
                # Forward pass
                data = self.model.training_step(batch, self.writer)

                # Backward pass
                self.optimizer.zero_grad()
                data.loss.backward()
                self.optimizer.step()

                # Accuracy data
                train_correct += data.correct
                train_size += data.size

                # Log the loss
                train_loss += data.loss.item()
                # self.writer.add_scalar(
                #     "Loss/train", data.loss, epoch * len(self.train_loader) + i
                # )

            # Calculate the average loss and accuracy
            train_loss /= len(self.train_loader)
            train_accuracy = train_correct / train_size

            # Print the progress and write to tensorboard
            print(f"Epoch: {epoch + 1}/{EPOCHS}, Loss: {train_loss:7f}")
            self.writer.add_scalar("Loss/train", train_loss, epoch)

            print(f"Accuracy: {train_accuracy:.5f}")
            self.writer.add_scalar("Accuracy/train", train_accuracy, epoch)

            print(
                "----------------------------------------------------------------------------------"
            )

            print("Validating...")

            # Validate
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for _, val_batch in enumerate(self.val_loader):
                    if cuda_available:
                        val_batch = (val_batch[0].to("cuda"), val_batch[1].to("cuda"))
                    # Validate
                    # Forward pass
                    data = self.model.validation_step(val_batch, self.writer)
                    val_loss += data.loss.item()

                    # Accuracy data
                    correct += data.correct
                    total += data.size

            val_loss /= len(self.val_loader)

            val_accuracy = correct / total

            # Print the progress and write to tensorboard
            print(f"Epoch: {epoch + 1}/{EPOCHS}, Loss: {val_loss:.7f}")
            self.writer.add_scalar("Loss/val", val_loss, epoch)

            print(f"Accuracy: {val_accuracy:.5f}")
            self.writer.add_scalar("Accuracy/val", val_accuracy, epoch)

            # Early stopping
            if self.early_stop(val_loss, train_loss):
                break
            else:
                self.model.save(
                    f"models/cnn/cnn_{LEARNING_RATE}_{BATCH_SIZE}_{EPOCHS}_{MIN_DELTA}_{PATIENCE}_{NUM_CLASSES}_{self.id}{'_landmarks' if isinstance(self.model, CnnLandMarks) else ''}.pth"
                )

            print(
                "----------------------------------------------------------------------------------"
            )

        print("End of training.")


class StepData:
    def __init__(self, loss: Tensor, predicted: Tensor, correct: int, size: int):
        """
        Initializes the StepData class.

        Args:
            loss (Tensor): The loss value.
            predicted (Tensor): The predicted values.
            correct (int): The number of correct predictions.
            size (int): The size of the data.
        """
        self.loss = loss
        self.predicted = predicted
        self.correct = correct
        self.size = size

    def __str__(self) -> str:
        """
        Returns a string representation of the StepData object.

        Returns:
            str: The string representation of the StepData object.
        """
        return f"Loss: {self.loss:.7f}, Correct: {self.correct}/{self.size}"


class CNN(bm.BaseModel):
    def __init__(
        self, train_file: str, test_file: str, val_file: str, train: bool = True
    ):
        """
        Initializes the CNN class. (Pixel based CNN model)

        Args:
            train_file (str): The path to the training file.
            test_file (str): The path to the test file.
            val_file (str): The path to the validation file.
        """
        super().__init__(train_file, test_file, val_file, train=train)

        # Define the layers of your CNN here
        self.conv1 = nn.Conv2d(IN_CHANNELS, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 7x7

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 3x3

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1)
        self.act4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)  # 1x1

        self.dropout = nn.Dropout(0.15)
        self.flatten = nn.Flatten()

        # self.fc4 = nn.Linear(128 * 3 * 3, out_features=512)
        # self.bn4 = nn.BatchNorm1d(512)
        # self.act4 = nn.ReLU()
        # self.dropout4 = nn.Dropout(0.3)

        # self.fc5 = nn.Linear(512, 256)
        # self.bn5 = nn.BatchNorm1d(256)
        # self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(64 * 7 * 7, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.act6 = nn.ReLU()

        self.fc7 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.act7 = nn.ReLU()

        self.fc8 = nn.Linear(64, NUM_CLASSES)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass through the CNN model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.act3(x)
        # x = self.bn3(x)
        # x = self.dropout(x)
        # x = self.pool3(x)

        # x = self.conv4(x)
        # x = self.act4(x)
        # x = self.bn4(x)
        # x = self.dropout(x)
        # x = self.pool4(x)

        x = self.flatten(x)

        # x = self.fc4(x)
        # x = self.bn4(x)
        # x = self.act4(x)
        # x = self.dropout4(x)

        # x = self.fc5(x)
        # x = self.bn5(x)
        # x = self.act5(x)

        x = self.fc6(x)
        x = self.act6(x)
        x = self.bn6(x)
        x = self.dropout(x)

        x = self.fc7(x)
        x = self.act7(x)
        x = self.bn7(x)

        x = self.fc8(x)

        x = self.soft_max(x)

        return x

    def training_step(self, batch: Any, writer: SummaryWriter) -> StepData:
        """
        Performs a training step.

        Args:
            batch (Any): The input batch.
            writer (SummaryWriter): The SummaryWriter object for logging.

        Returns:
            StepData: The StepData object containing the loss and accuracy information.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)
        loss = func.cross_entropy(outputs, labels)

        # Log the images
        # grid = func.make_grid(images)
        # writer.add_image("images", grid)

        # Accuracy data
        _, predicted = torch.max(outputs, 1)
        size = labels.size(0)
        correct = (predicted == labels).sum().item()

        step_data = StepData(loss, predicted, correct, size)

        # return loss
        return step_data

    def validation_step(self, batch: Any, writer: SummaryWriter) -> StepData:
        """
        Performs a validation step.

        Args:
            batch (Any): The input batch.
            writer (SummaryWriter): The SummaryWriter object for logging.

        Returns:
            StepData: The StepData object containing the loss and accuracy information.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)
        loss = func.cross_entropy(outputs, labels)

        # # Log the images
        # grid = func.make_grid(images)
        # writer.add_image("images", grid)

        # Accuracy data
        _, predicted = torch.max(outputs, 1)
        size = labels.size(0)
        correct = (predicted == labels).sum().item()

        step_data = StepData(loss, predicted, correct, size)

        # return data
        return step_data

    def test_step(self, batch: Any) -> StepData:
        """
        Performs a test step.

        Args:
            batch (Any): The input batch.
            writer (SummaryWriter): The SummaryWriter object for logging.

        Returns:
            StepData: The StepData object containing the loss and accuracy information.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)
        loss = func.cross_entropy(outputs, labels)

        # Log the images
        # grid = func.make_grid(images)
        # writer.add_image("images", grid)

        # Accuracy data
        _, predicted = torch.max(outputs, 1)
        size = labels.size(0)
        correct = (predicted == labels).sum().item()

        step_data = StepData(loss, predicted, correct, size)

        # return data
        return step_data

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def train_model(self):
        """
        Trains the CNN model.
        """
        train_loader = DataLoader(
            self.train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            self.val_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        test_loader = DataLoader(
            self.test_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )

        trainer = Trainer(
            self,
            train_loader,
            val_loader,
            test_loader,
            torch.optim.Adam(self.parameters(), lr=LEARNING_RATE),
        )
        trainer.train_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the given data.

        :param x: The input data.

        :return: ndarray with the predicted values.
        """
        np_data = x

        # Convert to bgr from rgb
        np_data = cv2.cvtColor(np_data, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        np_data = cv2.cvtColor(np_data, cv2.COLOR_BGR2GRAY)

        # Resize the image to 28x28
        np_data = cv2.resize(np_data, (28, 28))

        # Convert to tensor
        np_data = torch.tensor(np_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        dataset = data.TensorDataset(np_data)  # Create a TensorDataset from the data
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE
        )  # Use the dataset in the DataLoader

        self.eval()
        for batch in loader:
            # There is only one batch
            return self(batch[0]).detach().numpy()[0]
        return np.zeros(6)

    def test(self):
        """
        Tests the CNN model.
        """
        test_loader = DataLoader(self.test_data, batch_size=BATCH_SIZE, shuffle=True)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                # Forward pass
                data = self.test_step(batch)
                # Accuracy data
                correct += data.correct
                total += data.size

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.5f}")

    def visualize(self, data: pd.DataFrame, num_images: int = 5):
        """
        Visualizes the given data.

        Args:
            data (pd.DataFrame): The input data.
            num_images (int, optional): The number of images to visualize. Defaults to 5.
        """
        data = data.to_numpy().reshape(-1, 28, 28)

        for i in range(num_images):
            plt.imshow(data[i], cmap="gray")
            plt.show()

            img = Image.fromarray(data[i])
            img = ToTensor()(img)
            img = img.unsqueeze(0)

            self.eval()
            y_hat = self(img)
            print(f"Predicted: {torch.argmax(y_hat)}")

            print("Actual:")
            plt.imshow(data[i], cmap="gray")
            plt.show()

            print("Predicted:")
            plt.imshow(data[i], cmap="gray")
            plt.show()

            print("---------------------------------")

    def save(self, path: str):
        """
        Saves the model to the given path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        Loads the model from the given path.

        Args:
            path (str): The path to load the model from.

        Returns:
            CNN: The loaded CNN model.
        """
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path), strict=False)
        else:
            self.load_state_dict(
                torch.load(path, map_location=torch.device("cpu")), strict=False
            )
        self.eval()

    def __str__(self) -> str:
        """
        Returns a string representation of the CNN model.

        Returns:
            str: The string representation of the CNN model.
        """
        return "CNN"

    def __repr__(self) -> str:
        """
        Returns a string representation of the CNN model.

        Returns:
            str: The string representation of the CNN model.
        """
        return "CNN"

    def __len__(self) -> int:
        """
        Returns the length of the training data.

        Returns:
            int: The length of the training data.
        """
        return len(self.train_data)


class CnnLandMarks(bm.BaseModel):
    """Cnn model for landmarks detection."""

    def __init__(
        self, train_file: str, test_file: str, val_file: str, train: bool = True
    ):
        """
        Initializes the CNN class.

        Args:
            train_file (str): The path to the training file.
            test_file (str): The path to the test file.
            val_file (str): The path to the validation file.
        """
        super().__init__(train_file, test_file, val_file, landmarks=True, train=train)

        self.conv1 = nn.Conv2d(IN_CHANNELS, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)  # 14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)  # 7x7

        # self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        # self.act3 = nn.ReLU()
        # self.bn3 = nn.BatchNorm2d(128)
        # self.dropout3 = nn.Dropout(0.1)
        # self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(768, 256)
        self.act4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(256)

        # self.dropout4 = nn.Dropout(0.2)

        # self.fc5 = nn.Linear(512, 256)
        # self.act5 = nn.ReLU()
        # self.bn5 = nn.BatchNorm1d(256)

        self.fc6 = nn.Linear(256, 128)
        self.act6 = nn.ReLU()
        self.bn6 = nn.BatchNorm1d(128)

        self.fc7 = nn.Linear(128, 64)
        self.act7 = nn.ReLU()
        self.bn7 = nn.BatchNorm1d(64)

        self.fc8 = nn.Linear(64, NUM_CLASSES)

        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs a forward pass through the CNN model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.dropout1(x)
        x = self.pool2(x)

        # x = self.conv3(x)
        # x = self.act3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        # x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc4(x)
        x = self.act4(x)
        # x = self.bn4(x)

        # x = self.dropout2(x)

        # x = self.fc5(x)
        # x = self.act5(x)
        # x = self.bn5(x)

        x = self.dropout2(x)

        x = self.fc6(x)
        x = self.act6(x)
        # x = self.bn6(x)

        x = self.dropout2(x)

        x = self.fc7(x)
        x = self.act7(x)
        # x = self.bn7(x)

        x = self.dropout2(x)

        x = self.fc8(x)

        x = self.soft_max(x)

        return x

    def training_step(self, batch: Any, writer: SummaryWriter) -> StepData:
        """
        Performs a training step.

        Args:
            batch (Any): The input batch.
            writer (SummaryWriter): The SummaryWriter object for logging.

        Returns:
            StepData: The StepData object containing the loss and accuracy information.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)
        loss = func.cross_entropy(outputs, labels)

        # Log the images
        # grid = func.make_grid(images)
        # writer.add_image("images", grid)

        # Accuracy data
        _, predicted = torch.max(outputs, 1)
        size = labels.size(0)
        correct = (predicted == labels).sum().item()

        step_data = StepData(loss, predicted, correct, size)

        # return loss
        return step_data

    def validation_step(self, batch: Any, writer: SummaryWriter) -> StepData:
        """
        Performs a validation step.

        Args:
            batch (Any): The input batch.
            writer (SummaryWriter): The SummaryWriter object for logging.

        Returns:
            StepData: The StepData object containing the loss and accuracy information.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)
        loss = func.cross_entropy(outputs, labels)

        # # Log the images
        # grid = func.make_grid(images)
        # writer.add_image("images", grid)

        # Accuracy data
        _, predicted = torch.max(outputs, 1)
        size = labels.size(0)
        correct = (predicted == labels).sum().item()

        step_data = StepData(loss, predicted, correct, size)

        # return data
        return step_data

    def test_step(self, batch: Any, writer: SummaryWriter) -> StepData:
        """
        Performs a test step.

        Args:
            batch (Any): The input batch.
            writer (SummaryWriter): The SummaryWriter object for logging.

        Returns:
            StepData: The StepData object containing the loss and accuracy information.
        """
        # Forward pass
        images, labels = batch
        outputs = self(images)
        loss = func.cross_entropy(outputs, labels)

        # Log the images
        # grid = func.make_grid(images)
        # writer.add_image("images", grid)

        # Accuracy data
        _, predicted = torch.max(outputs, 1)
        size = labels.size(0)
        correct = (predicted == labels).sum().item()

        step_data = StepData(loss, predicted, correct, size)

        # return data
        return step_data

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        return torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def train_model(self):
        """
        Trains the CNN model.
        """
        train_loader = DataLoader(self.train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=BATCH_SIZE, shuffle=True)

        trainer = Trainer(
            self,
            train_loader,
            val_loader,
            test_loader,
            torch.optim.Adam(self.parameters(), lr=LEARNING_RATE),
        )
        trainer.train_model()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Makes predictions on the given data.

        :param x: The input data.

        :return: ndarray with the predicted values.
        """
        mp_settings = MpSettings()
        # Detect the landmarks
        result = mp_settings.hands.process(x)

        if not result:
            print("No result")
            return np.zeros(24)
        if not result.multi_hand_landmarks:
            print("No landmarks detected")
            return np.zeros(24)
        landmarks = result.multi_hand_landmarks[0]

        landmarks = np.array(
            [[landmark.x, landmark.y] for landmark in landmarks.landmark]
        ).flatten()

        landmarks = landmarks.reshape(1, 21, 2)
        # Convert to tensor
        np_data = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)

        dataset = data.TensorDataset(np_data)  # Create a TensorDataset from the data
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)

        self.eval()
        for batch in loader:
            # There is only one batch
            return self(batch[0]).detach().numpy()[0]

        return np.zeros(24)

    def test(self):
        """
        Tests the CNN model.
        """
        test_loader = DataLoader(self.test_data, batch_size=BATCH_SIZE, shuffle=True)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, batch in enumerate(test_loader):
                # Forward pass
                data = self.test_step(batch, SummaryWriter())
                # Accuracy data
                correct += data.correct
                total += data.size

        accuracy = correct / total
        print(f"Accuracy: {accuracy:.5f}")

    def visualize(self, data: pd.DataFrame, num_images: int = 5):
        """
        Visualizes the given data.

        Args:
            data (pd.DataFrame): The input data.
            num_images (int, optional): The number of images to visualize. Defaults to 5.
        """
        data = data.to_numpy().reshape(-1, 28, 28)

        for i in range(num_images):
            plt.imshow(data[i], cmap="gray")
            plt.show()

            img = Image.fromarray(data[i])
            img = ToTensor()(img)
            img = img.unsqueeze(0)

            self.eval()
            y_hat = self(img)
            print(f"Predicted: {torch.argmax(y_hat)}")

            print("Actual:")
            plt.imshow(data[i], cmap="gray")
            plt.show()

            print("Predicted:")
            plt.imshow(data[i], cmap="gray")
            plt.show()

            print("---------------------------------")

    def save(self, path: str):
        """
        Saves the model to the given path.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """
        Loads the model from the given path.

        Args:
            path (str): The path to load the model from.

        Returns:
            CNN: The loaded CNN model.
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def __str__(self) -> str:
        """
        Returns a string representation of the CNN model.

        Returns:
            str: The string representation of the CNN model.
        """
        return "CNN_21_landmarks"

    def __repr__(self) -> str:
        """
        Returns a string representation of the CNN model.

        Returns:
            str: The string representation of the CNN model.
        """
        return "CNN_21_landmarks"

    def __len__(self) -> int:
        """
        Returns the length of the training data.

        Returns:
            int: The length of the training data.
        """
        return len(self.train_data)
