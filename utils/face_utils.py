# utils/face_utils.py
import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define a simple CNN model architecture (similar to FER2013 small models)
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*48*48, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# List of possible emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load pre-trained model (simulate with random weights for now)
def load_emotion_model():
    model = EmotionCNN()
    # In real case: model.load_state_dict(torch.load('models/emotion_cnn_model.pth'))
    model.eval()
    return model

# Predict emotion from uploaded face image
def predict_emotion(model, image_file):
    transforms.Compose([
    transforms.Resize((224, 224)),  # fixed size
    transforms.ToTensor()

    ])

    image = Image.open(image_file).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # batch dimension

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    emotion = emotion_labels[predicted.item()]
    return emotion
