import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
from PIL import Image
import csv
import numpy as np
import json
import time
from pathlib import Path
import sys
import argparse

# =========================================
# Step 1: Device Configuration
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================
# Step 2: Data Preprocessing & Loading
# =========================================
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 classes - define early for CustomImageDataset
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# Custom Dataset for uploaded images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images from uploads directory
        for img_path in self.image_dir.glob("*.jpg") + self.image_dir.glob("*.jpeg") + self.image_dir.glob("*.png") + self.image_dir.glob("*.webp"):
            self.images.append(img_path)
            # For custom training, we'll use a dummy label or try to infer from filename
            label = self.infer_label_from_filename(img_path.name)
            self.labels.append(label)
    
    def infer_label_from_filename(self, filename):
        """Try to infer label from filename, otherwise use 'custom'"""
        filename_lower = filename.lower()
        for class_name in CLASSES:
            if class_name in filename_lower:
                return CLASSES.index(class_name)
        return 0  # Default to first class if no match
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = train_dataset.classes
print("Classes:", classes)

# =========================================
# Step 3: Load Pretrained Model
# =========================================
def create_model():
    """Create ResNet18 model for CIFAR-10"""
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

model = create_model().to(device)

# Try to load fine-tuned model if exists
try:
    if os.path.exists("fine_tuned_EEG_CIFAR10.pth"):
        model.load_state_dict(torch.load("fine_tuned_EEG_CIFAR10.pth", map_location=device))
        print("Loaded fine-tuned model successfully.")
    else:
        print("No fine-tuned model found. Using fresh ResNet18.")
except Exception as e:
    print(f"⚠️ Could not load model: {e}. Using fresh ResNet18.")

# =========================================
# Step 4: Define Loss & Optimizer
# =========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =========================================
# Training progress tracking
# =========================================
def save_progress(epoch, total_epochs, loss, accuracy=None, mode="dataset"):
    """Save training progress for Flask app to read"""
    progress = {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "loss": loss,
        "accuracy": accuracy,
        "progress_percent": int((epoch / total_epochs) * 100),
        "mode": mode,
        "timestamp": time.time()
    }
    with open("training_progress.json", "w") as f:
        json.dump(progress, f)

def save_training_epoch_to_csv(epoch, loss, accuracy, mode="dataset"):
    """Save training epoch data to CSV for live charting"""
    os.makedirs("results", exist_ok=True)
    csv_path = "results/training_results.csv"
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header only if file is new
        if not file_exists:
            writer.writerow(["Epoch", "Training Loss", "Training Accuracy", "Mode"])
        writer.writerow([epoch, loss, accuracy, mode])

def update_training_summary(epoch, total_epochs, loss, accuracy, mode="dataset"):
    """Update summary file incrementally during training"""
    os.makedirs("results", exist_ok=True)
    summary_path = "results/summary.txt"
    
    with open(summary_path, "w") as f:
        f.write(f"Model: ResNet18 (fine-tuned on {mode} images)\n")
        f.write(f"Current Accuracy: {accuracy:.2f}%\n")
        f.write(f"Current Loss: {loss:.4f}\n")
        f.write(f"Epoch: {epoch}/{total_epochs}\n")
        f.write(f"Training Mode: {mode}\n")
        f.write("Optimizer: Adam\nLoss Function: CrossEntropyLoss\n")

# =========================================
# Main Training Functions
# =========================================
def train_on_dataset(epochs=5):
    """Train on CIFAR-10 dataset"""
    print("Starting training on CIFAR-10 dataset...\n")
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # Save progress for Flask app
        save_progress(epoch + 1, epochs, avg_loss, accuracy, "dataset")
        # Save to CSV for live charting
        save_training_epoch_to_csv(epoch + 1, avg_loss, accuracy, "dataset")
        # Update summary with current progress
        update_training_summary(epoch + 1, epochs, avg_loss, accuracy, "dataset")
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_losses, train_accuracies

def train_on_custom_images(epochs=3, custom_image_path=None, custom_label=None):
    """Train on custom uploaded images or a single custom image with label"""
    
    # If single custom image provided, train with that
    if custom_image_path and custom_label:
        print(f"Fine-tuning with single image: {custom_image_path}, Label: {custom_label}\n")
        try:
            # Load single image
            img = Image.open(custom_image_path).convert('RGB')
            img_tensor = transform_train(img).unsqueeze(0)
            
            # Get label index
            label_idx = CLASSES.index(custom_label.lower())
            label_tensor = torch.tensor([label_idx])
            
            train_losses = []
            train_accuracies = []
            
            for epoch in range(epochs):
                model.train()
                
                # Train on this single image multiple times (repeat it)
                for _ in range(10):  # Train for 10 iterations per epoch
                    images = img_tensor.to(device)
                    labels = label_tensor.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    outputs = model(img_tensor.to(device))
                    _, predicted = torch.max(outputs, 1)
                    accuracy = 100.0 if predicted.item() == label_idx else 0.0
                    loss = criterion(outputs, label_tensor.to(device)).item()
                
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                
                save_progress(epoch + 1, epochs, loss, accuracy, "custom_image")
                save_training_epoch_to_csv(epoch + 1, loss, accuracy, "custom_image")
                update_training_summary(epoch + 1, epochs, loss, accuracy, "custom_image")
                
                print(f"Custom Image Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            return train_losses, train_accuracies
        except Exception as e:
            print(f"Error training on custom image: {e}")
            return [], []
    
    # Otherwise train on all custom images in uploads folder
    print("Starting training on custom uploaded images...\n")
    
    # Create custom dataset from uploads
    custom_dataset = CustomImageDataset("static/uploads", transform=transform_train)
    
    if len(custom_dataset) == 0:
        print("No custom images found for training!")
        return [], []
    
    custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)
    
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(custom_loader, desc=f"Custom Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(custom_loader)
        accuracy = 100 * correct / total if total > 0 else 0
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        # Save progress for Flask app
        save_progress(epoch + 1, epochs, avg_loss, accuracy, "custom")
        # Save to CSV for live charting
        save_training_epoch_to_csv(epoch + 1, avg_loss, accuracy, "custom")
        # Update summary with current progress
        update_training_summary(epoch + 1, epochs, avg_loss, accuracy, "custom")
        
        print(f"Custom Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return train_losses, train_accuracies

def train_model(epochs=5, mode="dataset", train_on_custom=False, custom_image_path=None, custom_label=None):
    """Main training function that supports both dataset and custom training"""
    
    # Clear old CSV to start fresh
    csv_path = "results/training_results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    # If custom image provided, use it for a quick fine-tune
    if train_on_custom and custom_image_path and custom_label:
        print(f"\n[TRAINING] Fine-tuning with custom image: {custom_label}")
        train_losses, train_accuracies = train_on_custom_images(epochs=2, custom_image_path=custom_image_path, custom_label=custom_label)
    elif mode == "dataset":
        train_losses, train_accuracies = train_on_dataset(epochs)
    elif mode == "custom":
        train_losses, train_accuracies = train_on_custom_images(epochs)
    else:
        print(f"Unknown training mode: {mode}")
        return 0.0

    print("Training complete [OK]")

    # =========================================
    # Evaluate the Model
    # =========================================
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_accuracy = 100 * correct / total
    print(f"\nModel Accuracy on CIFAR-10 test set: {final_accuracy:.2f}%")

    # =========================================
    # Save Fine-Tuned Model
    # =========================================
    torch.save(model.state_dict(), "fine_tuned_EEG_CIFAR10.pth")
    print("Fine-tuned model saved as fine_tuned_EEG_CIFAR10.pth ✅")

    # =========================================
    # Save Training Results
    # =========================================
    os.makedirs("results", exist_ok=True)
    
    # Save loss and accuracy data
    with open("results/training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Training Loss", "Training Accuracy", "Mode"])
        for i, (loss, acc) in enumerate(zip(train_losses, train_accuracies), start=1):
            writer.writerow([i, loss, acc, mode])

    with open("results/summary.txt", "w") as f:
        f.write(f"Model: ResNet18 (fine-tuned on {mode} images)\n")
        f.write(f"Final Test Accuracy: {final_accuracy:.2f}%\n")
        f.write(f"Epochs Trained: {epochs}\n")
        f.write(f"Training Mode: {mode}\n")
        f.write("Optimizer: Adam\nLoss Function: CrossEntropyLoss\n")

    # =========================================
    # Save Graphs
    # =========================================
    if train_losses and train_accuracies:
        try:
            # Create explicit axes so we can read the exact tick locations matplotlib chose
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].plot(range(1, len(train_losses)+1), train_losses, marker='o', color='blue', label='Training Loss')
            axes[0].set_title(f"Training Loss Curve ({mode})")
            axes[0].set_xlabel("Epochs")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(range(1, len(train_accuracies)+1), train_accuracies, marker='o', color='green', label='Training Accuracy')
            axes[1].set_title(f"Training Accuracy Curve ({mode})")
            axes[1].set_xlabel("Epochs")
            axes[1].set_ylabel("Accuracy (%)")
            axes[1].legend()
            axes[1].grid(True)

            plt.tight_layout()
            out_path = Path("results") / "training_curves.png"
            plt.savefig(str(out_path))

            # Extract tick locations and axis limits to save exact metadata for frontend
            try:
                loss_yticks = [float(x) for x in axes[0].get_yticks()]
                acc_yticks = [float(x) for x in axes[1].get_yticks()]
                loss_ylim = [float(v) for v in axes[0].get_ylim()]
                acc_ylim = [float(v) for v in axes[1].get_ylim()]
                xticks = [float(x) for x in axes[0].get_xticks()]

                meta = {
                    "loss_ticks": loss_yticks,
                    "acc_ticks": acc_yticks,
                    "loss_ylim": loss_ylim,
                    "acc_ylim": acc_ylim,
                    "xticks": xticks
                }
                # Write metadata JSON
                import json
                with open(Path('results') / 'chart_meta.json', 'w') as mf:
                    json.dump(meta, mf)
            except Exception as me:
                print(f"[WARN] could not save chart metadata: {me}")

            plt.close(fig)
            print("📊 Training graphs saved as 'results/training_curves.png'.")
        except Exception as e:
            print(f"[WARN] error saving training curves: {e}")

    # =========================================
    # Generate Prediction Grid (Sample predictions)
    # =========================================
    try:
        model.eval()
        
        # Get 9 random samples from test set
        sample_images = []
        sample_labels = []
        sample_predictions = []
        
        batch_count = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(min(3, len(images))):
                    if len(sample_images) < 9:
                        sample_images.append(images[i].cpu())
                        sample_labels.append(CLASSES[labels[i].item()])
                        sample_predictions.append(CLASSES[predicted[i].item()])
                
                if len(sample_images) >= 9:
                    break
        
        # Create grid visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Sample Predictions from Test Set', fontsize=16, fontweight='bold')
        
        grid_meta = []
        for idx, (ax, img, true_label, pred_label) in enumerate(zip(axes.flat, sample_images, sample_labels, sample_predictions)):
            # Denormalize image
            img = img * 0.5 + 0.5  # Reverse normalization
            img = img.clamp(0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            ax.imshow(img)
            
            # Color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontweight='bold')
            # record metadata for this grid cell
            grid_meta.append({
                'index': idx,
                'true': true_label,
                'pred': pred_label
            })
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig("results/predictions_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("[OK] Prediction grid saved as 'results/predictions_grid.png' (300 DPI, high quality).")
        
        # Save grid metadata so frontend can show per-sample true/pred labels
        try:
            import json
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'predictions_grid_meta.json'), 'w') as gm:
                json.dump({'cells': grid_meta}, gm, indent=2)
            print(f"[OK] Prediction metadata saved ({len(grid_meta)} samples)")
            for i, cell in enumerate(grid_meta):
                print(f"     Sample {i}: {cell['true']} -> {cell['pred']}")
        except Exception as e:
            print(f"[WARN] could not write predictions grid meta: {e}")
    except Exception as e:
        print(f"⚠️ Could not generate prediction grid: {e}")

    # Clear progress file after training
    if os.path.exists("training_progress.json"):
        os.remove("training_progress.json")

    return final_accuracy

# =========================================
# Prediction Function
# =========================================
def predict_image(image_path):
    """Predict class and confidence for a single image"""
    model.eval()
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = classes[pred.item()]
    confidence = conf.item() * 100
    return label, confidence

# =========================================
# Standalone Execution
# =========================================
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train EEG-inspired classification model")
    parser.add_argument("--mode", type=str, default="dataset", choices=["dataset", "custom"],
                       help="Training mode: 'dataset' for CIFAR-10 or 'custom' for uploaded images")
    args = parser.parse_args()
    
    # Train with specified mode
    acc = train_model(epochs=5 if args.mode == "dataset" else 3, mode=args.mode)
    print(f"\nFinal Accuracy: {acc:.2f}%")