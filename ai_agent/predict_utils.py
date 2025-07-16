import torch
from torchvision import transforms
from PIL import Image
import cv2
from ai_agent.config_agent import MODEL_PATH

# Load model once
def load_model():
    from model import build_model
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return "Fake" if predicted.item() == 1 else "Real"

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = 0
    fake = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
        if predicted.item() == 1:
            fake += 1

    cap.release()
    return f"Fake Frames: {fake}/{total} ({100*fake/total:.2f}%)" if total > 0 else "No frames read"
