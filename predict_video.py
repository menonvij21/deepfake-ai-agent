import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import load_model

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("outputs/checkpoints/mobilenetv2.pt")
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return "FAKE" if predicted.item() == 1 else "REAL"

# Load video
input_path = input("🎥 Enter video path: ").strip('"')
cap = cv2.VideoCapture(input_path)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_prediction.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    label = predict_frame(frame)
    color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)
    cv2.putText(frame, f"Prediction: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    out.write(frame)

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
print("✅ Saved labeled output video as: output_prediction.mp4")
####env C:\Users\ASUS\venv310\Scripts\Activate.
