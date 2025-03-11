import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ROIPredictionNetwork(nn.Module):
    def __init__(self):
        super(ROIPredictionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(40 * 25 + 4, 100)
        self.fc2 = nn.Linear(100, 32)
        self.fc3 = nn.Linear(32, 4)
    
    def forward(self, x, prev_roi):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool3(x)
        
        x = torch.flatten(x, start_dim=1)
        x = torch.cat((x, prev_roi), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_event_map(frame_t, frame_t1, threshold=0.3):
    abs_diff = np.abs(frame_t.astype(np.float32) - frame_t1.astype(np.float32))
    norm_diff = abs_diff / (frame_t.astype(np.float32) + 1e-6)
    event_map = (norm_diff > threshold).astype(np.uint8) * 255
    return event_map

def extract_edge_map(segmentation_map):
    edges = cv2.Canny(segmentation_map, 100, 200)
    return edges

def predict_roi(model, frame_t, frame_t1, segmentation_map, prev_roi, device='cpu'):
    event_map = generate_event_map(frame_t, frame_t1)
    edge_map = extract_edge_map(segmentation_map)
    
    event_map = cv2.resize(event_map, (160, 100))
    edge_map = cv2.resize(edge_map, (160, 100))
    
    input_tensor = np.stack([event_map, edge_map], axis=0) / 255.0
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)
    prev_roi_tensor = torch.tensor(prev_roi, dtype=torch.float32).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        roi = model(input_tensor, prev_roi_tensor)
    
    return roi.cpu().numpy().flatten()

def visualize_roi(frame, roi, title="Predicted ROI"):
    x_min, y_min, x_max, y_max = map(int, roi)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(frame_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    plt.figure(figsize=(6,6))
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ROIPredictionNetwork().to(device)

frame_t = cv2.imread("frame_t.png", cv2.IMREAD_GRAYSCALE)
frame_t1 = cv2.imread("frame_t1.png", cv2.IMREAD_GRAYSCALE)
segmentation_map = cv2.imread("seg_map.png", cv2.IMREAD_GRAYSCALE)
prev_roi = [0, 0, frame_t.shape[1], frame_t.shape[0]]

predicted_roi = predict_roi(model, frame_t, frame_t1, segmentation_map, prev_roi, device)
print("Predicted ROI:", predicted_roi)
visualize_roi(frame_t1, predicted_roi)
