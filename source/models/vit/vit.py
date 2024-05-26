import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CSVImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class VIT:
    def __init__(
        self, csv_file, num_classes, batch_size=32, learning_rate=0.001, num_epochs=10
    ):
        self.csv_file = csv_file
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224)
                ),  # ViT typically requires a specific image size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.dataset = CSVImageDataset(csv_file, transform=self.transform)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model.head = nn.Linear(
            self.model.head.in_features, self.num_classes
        )  # Replace the head with the number of classes in your dataset
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for images, labels in self.dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Loss: {running_loss/len(self.dataloader)}"
            )

        print("Finished Training")

    def save_model(self, path="vit_model.pth"):
        torch.save(self.model.state_dict(), path)
        
    def predict(self, img: Image.Image) -> int:
        img = self.transform(img).unsqueeze(0).to(self.device)
        output = self.model(img)
        _, predicted = torch.max(output, 1)
        return predicted.item()
