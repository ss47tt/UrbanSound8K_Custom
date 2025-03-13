import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Define Dataset Class
class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = os.listdir(data_path)  # List subdirectories (classes)
        self.files = []
        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(data_path, class_name)
            for file_name in os.listdir(class_path):
                if file_name.endswith('.npy'):
                    self.files.append((class_name, os.path.join(class_path, file_name)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        class_name, file_path = self.files[idx]
        data = np.load(file_path)  # Load the .npy file (spectrogram)
        label = self.classes.index(class_name)

        if self.transform:
            data = self.transform(data)

        return data, label

# Define transformations
transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x).unsqueeze(0).repeat(3, 1, 1)),  # Convert numpy array to tensor and repeat the channel
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet's standard normalization
])

# Paths to training/validation and testing data
train_val_path = 'train_val_spectrogram_npy'  # Replace with your training/validation data path
test_path = 'test_spectrogram_npy'  # Replace with your test data path

# Load the dataset
train_val_dataset = CustomDataset(data_path=train_val_path, transform=transform)

# Split into train and validation
train_size = int(0.9 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load test dataset
test_dataset = CustomDataset(data_path=test_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Using the recommended weights argument
model.fc = nn.Linear(model.fc.in_features, 8)  # 8 classes

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {correct/total*100:.2f}%")

    # Validate the model
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    print(f"Validation Accuracy: {val_correct/val_total*100:.2f}%")

# Save the model after training is complete
torch.save(model.state_dict(), 'resnet50_spectrogram_model.pth')  # Save model weights

# # Test the model
# model.eval()
# test_correct = 0
# test_total = 0
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         test_correct += (predicted == labels).sum().item()
#         test_total += labels.size(0)

# print(f"Test Accuracy: {test_correct/test_total*100:.2f}%")