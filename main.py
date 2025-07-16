import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from config import *
from utils.dataloader import get_loaders
from models.mobilenetv2 import build_model
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = get_loaders()
model = build_model().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR)

# Create output path
os.makedirs("outputs/checkpoints", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(train_loader.dataset)
    print(f"Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "outputs/checkpoints/mobilenetv2.pt")
print("✅ Model saved to outputs/checkpoints/mobilenetv2.pt")
