import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import itertools
import seaborn as sns
import time
import pandas as pd
import cv2
import torchvision.transforms.functional as TF

# ---------------------
# Dataset Class Definition
# ---------------------
class EyeBoundingBoxDataset(Dataset):
    def __init__(self, subject_ids, root_dir, transform=None, apply_preprocessing=True, timing_enabled=False):
        self.root_dir = root_dir
        self.subject_ids = subject_ids
        self.transform = transform
        self.apply_preprocessing = apply_preprocessing
        self.timing_enabled = timing_enabled

        # For timing
        self.total_preprocessing_time = 0.0
        self.num_preprocessing_calls = 0

        self.data = []

        # Precompute Gamma LUT
        gamma = 0.8
        self.gamma_LUT = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)

        for subject_id in self.subject_ids:
            subject_dir = os.path.join(root_dir, 'openEDS', 'openEDS', subject_id)
            bbox_file = os.path.join(root_dir, 'bbox', 'bbox', f"{subject_id}.txt")
            with open(bbox_file, 'r') as f:
                bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]
            for i in range(len(bboxes) - 1):
                img0_path = os.path.join(subject_dir, f"{i}.png")
                img1_path = os.path.join(subject_dir, f"{i+1}.png")
                self.data.append((img0_path, img1_path, bboxes[i+1]))

    def __len__(self):
        return len(self.data)

    def preprocess_image(self, img_path):
        start_time = time.perf_counter() if self.timing_enabled else None

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image from: {img_path}")

        img = cv2.LUT(img, self.gamma_LUT)

        p_low, p_high = np.percentile(img, (1, 99))
        lut_indices = np.arange(256)
        stretch_LUT = np.clip((lut_indices - p_low) * (255.0 / max(p_high - p_low, 1)), 0, 255).astype(np.uint8)
        img = cv2.LUT(img, stretch_LUT)

        if self.timing_enabled:
            end_time = time.perf_counter()
            self.total_preprocessing_time += (end_time - start_time) * 1000  # ms
            self.num_preprocessing_calls += 1

        return img

    def __getitem__(self, idx):
        img0_path, img1_path, bbox = self.data[idx]

        if self.apply_preprocessing:
            img0 = self.preprocess_image(img0_path)
            img1 = self.preprocess_image(img1_path)
        else:
            img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

        # Resize manually because we skip torchvision transforms here
        img0 = cv2.resize(img0, (64, 64), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_LINEAR)

        img0 = torch.from_numpy(img0).unsqueeze(0).float() / 255.0
        img1 = torch.from_numpy(img1).unsqueeze(0).float() / 255.0

        diff = img1 - img0
        input_tensor = torch.cat((img1, diff), dim=0)

        target = torch.tensor(bbox, dtype=torch.float32)

        return input_tensor, target

    def get_average_preprocessing_time(self):
        if self.num_preprocessing_calls == 0:
            return 0.0
        return self.total_preprocessing_time / self.num_preprocessing_calls
    
# ---------------------
# Model Definition
# ---------------------
class LightweightBBoxCNN(nn.Module):
    def __init__(self, hidden_size=64):  # now configurable
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_size),  # updated hidden size
            nn.ReLU(),
            nn.Linear(hidden_size, 4)  # 4 = xmin, xmax, ymin, ymax
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# ---------------------
# Train-Test Split (Saves .txt File with Subject Names for Each Split)
# ---------------------   
def split_subjects_and_save(root_dir, output_file='subject_split.txt', seed=42):
    random.seed(seed)
    subject_path = os.path.join(root_dir, 'openEDS', 'openEDS')
    all_subjects = sorted([d for d in os.listdir(subject_path) if d.startswith('S_') and os.path.isdir(os.path.join(subject_path, d))])

    random.shuffle(all_subjects)
    n_total = len(all_subjects)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)

    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train:n_train + n_val]
    test_subjects = all_subjects[n_train + n_val:]

    with open(output_file, 'w') as f:
        f.write("Training Subjects:\n")
        for s in train_subjects:
            f.write(f"{s}\n")
        f.write("\nValidation Subjects:\n")
        for s in val_subjects:
            f.write(f"{s}\n")
        f.write("\nTest Subjects:\n")
        for s in test_subjects:
            f.write(f"{s}\n")

    print(f"Subject split saved to {output_file}")
    return train_subjects, val_subjects, test_subjects

# ---------------------
# Random Augmentations
# ---------------------
class RandomAugmentations:
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, img, label):
        if random.random() < self.p:
            # Horizontal Flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                label = TF.hflip(label)
                print('Horizontal Flip')

            # Random Rotation
            if random.random() < self.p:
                angle = random.uniform(-10, 10)
                img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                label = TF.rotate(label.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
                print('Random Rotation')

            # Gaussian Blur (ONLY image)
            if random.random() < self.p:
                img = TF.gaussian_blur(img, kernel_size=7, sigma=random.uniform(2, 7))
                # Do NOT blur the label!
                print('Gaussian Blur')

            # # Random Translation
            # if random.random() < self.p:
            #     max_dx = 20
            #     max_dy = 20
            #     dx = random.randint(-max_dx, max_dx)
            #     dy = random.randint(-max_dy, max_dy)
            #     img = TF.affine(img, angle=0, translate=[dx, dy], scale=1, shear=[0, 0])
            #     label = TF.affine(label.unsqueeze(0), angle=0, translate=[dx, dy], scale=1, shear=[0, 0], interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
            #     print('Random Translation')

            # Image corruption with thin lines (ONLY image)
            if random.random() < self.p:
                num_lines = random.randint(2, 9)
                img = self.draw_random_lines(img, num_lines)
                print('Ranodm Lines')

        return img, label

    def draw_random_lines(self, x, num_lines):
        _, h, w = x.shape
        center_x = random.randint(0, w-1)
        center_y = random.randint(0, h-1)

        img_np = (x.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        for _ in range(num_lines):
            angle = random.uniform(0, 360)
            length = random.randint(10, 100)
            x1 = int(center_x + length * np.cos(np.deg2rad(angle)))
            y1 = int(center_y + length * np.sin(np.deg2rad(angle)))
            x1 = np.clip(x1, 0, w-1)
            y1 = np.clip(y1, 0, h-1)
            cv2.line(img_np, (center_x, center_y), (x1, y1), (255,), thickness=1)
        img_np = img_np / 255.0
        return torch.from_numpy(img_np).unsqueeze(0).float()
    
# ---------------------
# New DataLoader
# ---------------------
class DynamicCroppedSegmentationDataset(Dataset):
    def __init__(self, subject_ids, bbox_model, root_dir, device='cpu', transform=None):
        self.root_dir = root_dir
        self.subject_ids = subject_ids
        self.device = device
        self.transform = transform
        self.bbox_model = bbox_model.to(device)
        self.bbox_model.eval()

        self.data = []
        for subject_id in subject_ids:
            subject_dir = os.path.join(self.root_dir, 'openEDS', 'openEDS', subject_id)
            if not os.path.exists(subject_dir):
                continue
            image_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.png')])
            for img_idx, img_file in enumerate(image_files):
                img_path = os.path.join(subject_dir, img_file)
                label_path = os.path.join(subject_dir, img_file.replace('.png', '.npy'))
                if os.path.exists(label_path):
                    self.data.append((subject_id, img_idx, img_path, label_path))

    def preprocess_image(self, img):
        gamma = 0.8
        gamma_LUT = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, gamma_LUT)

        p_low, p_high = np.percentile(img, (1, 99))
        lut_indices = np.arange(256)
        stretch_LUT = np.clip((lut_indices - p_low) * (255.0 / max(p_high - p_low, 1)), 0, 255).astype(np.uint8)
        img = cv2.LUT(img, stretch_LUT)

        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subject_id, img_idx, img_path, label_path = self.data[idx]

        img1 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        label = np.load(label_path)

        if img1 is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")

        img1 = self.preprocess_image(img1)
        h, w = img1.shape

        if img_idx == 0:
            # Frame 0: no cropping, full preprocessed image
            img_tensor = torch.from_numpy(img1).unsqueeze(0).float() / 255.0
            label_tensor = torch.from_numpy(label).long()

            if self.transform:
                img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

            return img_tensor, label_tensor

        else:
            # Frame > 0: use img0 and img1 to compute diff
            img0_path = os.path.join(self.root_dir, 'openEDS', 'openEDS', subject_id, f"{img_idx-1}.png")
            img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
            if img0 is None:
                raise FileNotFoundError(f"Cannot load previous image: {img0_path}")

            img0 = self.preprocess_image(img0)

            # Resize both for bbox model input (but ONLY for bbox prediction, not final crop!)
            img0_resized = cv2.resize(img0, (64, 64), interpolation=cv2.INTER_LINEAR)
            img1_resized = cv2.resize(img1, (64, 64), interpolation=cv2.INTER_LINEAR)

            img0_tensor = torch.from_numpy(img0_resized).unsqueeze(0).float() / 255.0
            img1_tensor = torch.from_numpy(img1_resized).unsqueeze(0).float() / 255.0

            diff = img1_tensor - img0_tensor

            input_tensor = torch.cat((img1_tensor, diff), dim=0).unsqueeze(0).to(self.device)

            # Bounding Box Prediction
            with torch.no_grad():
                pred_bbox = self.bbox_model(input_tensor).cpu().squeeze(0)

            xmin, xmax, ymin, ymax = pred_bbox

            # Clamp
            xmin = int(max(0, min(w-1, xmin.item())))
            xmax = int(max(0, min(w-1, xmax.item())))
            ymin = int(max(0, min(h-1, ymin.item())))
            ymax = int(max(0, min(h-1, ymax.item())))

            if xmin >= xmax or ymin >= ymax:
                cropped_img = img1
                cropped_label = label
            else:
                cropped_img = img1[ymin:ymax, xmin:xmax]
                cropped_label = label[ymin:ymax, xmin:xmax]

            img_tensor = torch.from_numpy(cropped_img).unsqueeze(0).float() / 255.0
            label_tensor = torch.from_numpy(cropped_label).long()

            if self.transform:
                img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

            return img_tensor, label_tensor
    
# ---------------------
# Setup Everything
# ---------------------
def setup(root_dir, bbox_model_path, batch_size=32, device='cuda'):
    # Load pretrained bbox model
    bbox_model = LightweightBBoxCNN()
    bbox_model.load_state_dict(torch.load(bbox_model_path, map_location=device))

    # Split subjects
    train_subjects, val_subjects, test_subjects = split_subjects_and_save(root_dir)

    # Transforms
    train_transform = RandomAugmentations(p=1)
    val_transform = None

    train_dataset = DynamicCroppedSegmentationDataset(train_subjects, bbox_model, root_dir, device=device, transform=train_transform)
    val_dataset = DynamicCroppedSegmentationDataset(val_subjects, bbox_model, root_dir, device=device, transform=val_transform)

    return train_dataset, val_dataset, bbox_model

ROOT_DIR = r"C:\Users\omarh\OneDrive - Georgia Institute of Technology\openEDS2019"
BBOX_MODEL_PATH = r"C:\Users\omarh\Documents\GaTech\VisualTracking\scripts\preprocessing\checkpoints\ppaug_cp_e50"

train_dataset, val_dataset, bbox_model = setup(ROOT_DIR, BBOX_MODEL_PATH)

# Display 5 examples
for i in range(5):
    img_tensor, label_tensor = train_dataset[i]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor.squeeze(0).cpu(), cmap='gray')
    plt.title(f"Image {i}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(label_tensor.cpu(), cmap='jet')
    plt.title(f"Label {i}")
    plt.axis('off')

    plt.show()