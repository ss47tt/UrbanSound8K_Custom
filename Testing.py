import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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

# Path to test data
test_path = 'test_spectrogram_npy'  # Replace with your test data path

# Load the dataset
test_dataset = CustomDataset(data_path=test_path, transform=transform)

# Create DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize ResNet50 model
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Using the recommended weights argument
model.fc = nn.Linear(model.fc.in_features, 8)  # 8 classes

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load the saved model weights
model.load_state_dict(torch.load('resnet50_spectrogram_model.pth'))  # Load model weights

# Set the model to evaluation mode
model.eval()

# Initialize variables to collect true labels and predictions
all_labels = []
all_predictions = []

# Evaluate on test data
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())  # Collect true labels
        all_predictions.extend(predicted.cpu().numpy())  # Collect predicted labels

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_predictions)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(all_labels, all_predictions, target_names=test_dataset.classes)

# Print classification report
print("Classification Report:")
print(class_report)

# Save confusion matrix and classification report to files
# Save the confusion matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# Save the classification report to a text file
with open('classification_report.txt', 'w') as f:
    f.write(class_report)