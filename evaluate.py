# evaluate.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # ✅ for progress bar
from config import TEST_DIR 
from utils.dataset import DeepfakeDataset
from model import load_model

# ✅ Load the model from correct path
model = load_model('outputs/checkpoints/mobilenetv2.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("✅ Starting evaluation...")

# ✅ Match transform with training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ✅ Load test dataset
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
print(f"📁 Found {len(test_dataset)} test images.")

# ✅ DataLoader for evaluation
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

correct = 0
total = 0

# ✅ Inference with progress
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="🔍 Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# ✅ Final report
print(f"✅ Evaluation completed.")
print(f"🔢 Total samples evaluated: {total}")
print(f"🎯 Correct predictions: {correct}")
print(f"✅ Test Accuracy: {100 * correct / total:.2f}%")
