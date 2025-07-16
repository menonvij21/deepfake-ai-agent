import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def build_model():
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model
