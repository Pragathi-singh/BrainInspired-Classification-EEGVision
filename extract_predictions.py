"""
Extract predictions from CIFAR-10 test set and generate accurate metadata
This matches the actual predictions_grid.png that was generated during training
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import json
import os
from pathlib import Path

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_FILE = 'fine_tuned_EEG_CIFAR10.pth'
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 10)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False))
model.to(DEVICE)
model.eval()

# Load test set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_set = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

# Extract predictions for 9 samples (to match the grid)
sample_labels = []
sample_predictions = []

print("Extracting predictions from test set...")
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        for i in range(len(images)):
            if len(sample_labels) < 9:
                sample_labels.append(CLASSES[labels[i].item()])
                sample_predictions.append(CLASSES[predicted[i].item()])
        
        if len(sample_labels) >= 9:
            break

print(f"✓ Extracted {len(sample_labels)} samples")

# Create metadata matching the predictions grid
grid_meta = []
for idx in range(len(sample_labels)):
    grid_meta.append({
        'index': idx,
        'true': sample_labels[idx],
        'pred': sample_predictions[idx]
    })
    print(f"  Sample {idx}: True={sample_labels[idx]}, Pred={sample_predictions[idx]}")

# Save metadata
os.makedirs('results', exist_ok=True)
with open('results/predictions_grid_meta.json', 'w') as f:
    json.dump({'cells': grid_meta}, f, indent=2)

print("\n✓ Metadata saved to results/predictions_grid_meta.json")
