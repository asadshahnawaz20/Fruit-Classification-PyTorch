import os
import numpy as np
import cv2

# Path to your test image
image_path = "evaluation/test_image.png"  # change if your path is different

# Number of fake images you want in train/validation
num_train = 20
num_val = 5

# Load the original image
img = cv2.imread(image_path)
img = cv2.resize(img, (64, 64))  # resize to match your network input
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
img = img.transpose(2, 0, 1)  # shape (C, H, W)

# Create train data and labels
train_data = np.array([img for _ in range(num_train)], dtype=np.uint8)
train_labels = np.array([i % 4 for i in range(num_train)], dtype=np.int64)  # 4 fake classes

# Create validation data and labels
val_data = np.array([img for _ in range(num_val)], dtype=np.uint8)
val_labels = np.array([i % 4 for i in range(num_val)], dtype=np.int64)

# Save files in fruit_data folder
os.makedirs("fruit_data", exist_ok=True)
np.save("fruit_data/train_data.npy", train_data)
np.save("fruit_data/train_labels.npy", train_labels)
np.save("fruit_data/validation_data.npy", val_data)
np.save("fruit_data/validation_labels.npy", val_labels)

print("✅ Demo dataset created successfully!")
print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
