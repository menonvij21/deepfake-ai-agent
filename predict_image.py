import torch
from torchvision import transforms
from PIL import Image
import os
from model import load_model

# Load model
model = load_model('outputs/checkpoints/mobilenetv2.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Get and clean path
path = input("📷 Enter image path: ").strip()
path = os.path.normpath(path.strip('"').strip("'"))

try:
    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    label = "FAKE" if predicted.item() == 1 else "REAL"
    print(f"🧠 Prediction: {label.upper()}")

except FileNotFoundError:
    print("❌ File not found. Please check the path.")
except Exception as e:
    print(f"❌ Error: {e}")
