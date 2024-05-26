import numpy as np
import pandas as pd
import torch
import json
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BaseModel(nn.Module):
    """
    Basemodel for all ML models
    """
    def __init__(self, train_file: str, test_file: str, val_file: str, train: bool = True, landmarks: bool = False) -> None:
        super(BaseModel, self).__init__()

        if train:
            if landmarks:
                self.train_data = LandmarksDataset(train_file)
                self.test_data = LandmarksDataset(test_file)
                self.val_data = LandmarksDataset(val_file)
            else:
                self.train_data = ImageDataset(train_file)
                self.test_data = ImageDataset(test_file)
                self.val_data = ImageDataset(val_file)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass logic
        
        :param x: Input tensor
        :return: Output tensor
        """
        # Implement the forward pass of your base model here
        # This will be common for all models based on this base model
        return x
    
    def save(self, path: str) -> None:
        """
        Save the model to the path provided
        
        :param path: Path to save the model
        """
        torch.save(self.state_dict(), path)
                   
    def load(self, path: str) -> None:
        """
        Load the model from the path provided
        
        :param path: Path to load the model
        """
        self.load_state_dict(torch.load(path))
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predict the output for the input x
        """
        np_x = np.array(x)
        x_tensor = torch.from_numpy(np_x).float()  # Convert Image to Tensor
        return self.forward(x_tensor).numpy()
    
    def train_model(self):
        """ Train the model
        """
        pass
    
    def test(self):
        """ Test the model
        """
        pass
    
class ImageDataset(Dataset):
    """Dataset class for image data
    """
    def __init__(self, anotation_file: str, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(anotation_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        return len(self.img_labels)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        label = self.img_labels.iloc[idx, 0]
        pixels = self.img_labels.iloc[idx, 1:].values.astype('uint8').reshape((1, 28, 28))
        image = torch.from_numpy(pixels).float()  # Convert to float
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class LandmarksDataset(Dataset):
    """Dataset class for landmarks data
    """
    def __init__(self, anotation_file: str, transform=None, target_transform=None):
        self.landmarks = pd.read_csv(anotation_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        return len(self.landmarks)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Any]:
        label = self.landmarks.iloc[idx, 0]
        # Parse string into list of floats
        landmarks = self.landmarks.iloc[idx, 1:]
        landmarks = [json.loads(str(landmark)) for landmark in landmarks]
        # Convert the landmarks list into a tensor
        landmarks = torch.tensor(landmarks).float()
        landmarks = landmarks.unsqueeze(0)
        # Flatten the landmarks tensor
        # landmarks = landmarks.view(-1)
        if self.transform:
            landmarks = self.transform(landmarks)
        if self.target_transform:
            label = self.target_transform(label)
        return landmarks, label



class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    """
    def __init__(self, min_delta: float, patience: int) -> None:
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        
    def early_stop_check(self, val_loss: float, train_loss: float) -> bool:
        """Early stopping check

        Args:
            val_loss (float): Validation loss
            train_loss (float): Training loss

        Returns:
            bool: True if early stopping condition is met
        """
        if (val_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter > self.patience:  
                return True
        else:
            self.counter = 0
        return False
