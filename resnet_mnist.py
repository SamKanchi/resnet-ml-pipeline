
from pathlib import Path
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18


# ---------------------------------------------
# STEP 1: CONFIGURATION
# ---------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Device:", DEVICE)


# ---------------------------------------------
# STEP 2: LOAD MNIST DATASET
# ---------------------------------------------

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ---------------------------------------------
# STEP 3: CREATE RESNET18 MODEL
# ---------------------------------------------

model = resnet18(weights=None)

# Change RGB input to grayscale
# Keep small MNIST images at full resolution
model.conv1 = nn.Conv2d(
    in_channels=1,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)

# Remove the initial max-pooling operation
model.maxpool = nn.Identity()

# Change ImageNet's 1000 classes to 10
model.fc = nn.Linear(
    model.fc.in_features,
    10
)

model = model.to(DEVICE)

print("\nResNet18 created successfully")

total_params = sum(
    p.numel() for p in model.parameters()
)

print(f"Total parameters: {total_params:,}")


# ---------------------------------------------
# STEP 4: LOSS FUNCTION AND OPTIMIZER
# ---------------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------------
# STEP 5: VALIDATION FUNCTION
# ---------------------------------------------

def evaluate(data_loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    return avg_loss, accuracy


# ---------------------------------------------
# STEP 6: TRAINING LOOP
# ---------------------------------------------

train_losses = []
val_losses = []
val_accuracies = []

best_val_accuracy = -1.0

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Clear old gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    avg_train_loss = running_loss / len(train_dataset)

    val_loss, val_accuracy = evaluate(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train loss: {avg_train_loss:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.2%}",
        flush=True
    )

    # Save the best validation model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            OUTPUT_DIR / "best_resnet18.pth"
        )


# ---------------------------------------------
# STEP 7: EVALUATE ON TEST DATA
# ---------------------------------------------

# Reload the best validation checkpoint
model.load_state_dict(
    torch.load(
        OUTPUT_DIR / "best_resnet18.pth",
        map_location=DEVICE,
        weights_only=True
    )
)

test_loss, test_accuracy = evaluate(test_loader)

print(f"\nFinal test loss: {test_loss:.4f}")
print(f"Final test accuracy: {test_accuracy:.2%}")


# ---------------------------------------------
# STEP 8: TRAINING GRAPHS
# ---------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(train_losses, marker="o", label="Train loss")
plt.plot(val_losses, marker="o", label="Validation loss")

plt.xlabel("Epoch (starting at 0)")
plt.ylabel("Cross-entropy loss")
plt.title("ResNet18 Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "loss_curve.png")
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(val_accuracies, marker="o")

plt.xlabel("Epoch (starting at 0)")
plt.ylabel("Accuracy")
plt.title("ResNet18 Validation Accuracy")
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "accuracy_curve.png")
plt.close()


# ---------------------------------------------
# STEP 9: VISUALIZE TEST PREDICTIONS
# ---------------------------------------------

images, labels = next(iter(test_loader))

model.eval()

with torch.no_grad():
    predictions = model(
        images.to(DEVICE)
    ).argmax(dim=1).cpu()

fig, axes = plt.subplots(2, 5, figsize=(11, 5))

for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze(), cmap="gray")

    predicted = predictions[i].item()
    actual = labels[i].item()

    ax.set_title(
        f"Pred: {predicted} | True: {actual}"
    )
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions.png")
plt.close()


print("\nTraining complete!")
print("Saved files in:", OUTPUT_DIR)
