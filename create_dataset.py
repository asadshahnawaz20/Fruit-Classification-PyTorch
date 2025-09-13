import os
import numpy as np
import cv2

# Folder containing your images
image_folder = "evaluation"  # change if your images are somewhere else

# Lists to store image data and labels
data = []
labels = []

# Assign a numeric label to each subfolder (if multiple classes)
categories = [d for d in os.listdir(image_folder) if os.path.isdir(os.path.join(image_folder, d))]

if not categories:
    # If there are no subfolders, treat all images as one class
    categories = ["fruits"]

for idx, category in enumerate(categories):
    folder_path = os.path.join(image_folder, category)
    
    if not os.path.isdir(folder_path):
        # If single folder mode
        folder_path = image_folder
    
    for file in os.listdir(folder_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # resize to 64x64
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            data.append(img)
            labels.append(idx)

# Convert lists to numpy arrays
data = np.array(data, dtype=np.uint8)
labels = np.array(labels, dtype=np.int64)

# Transpose to match model input: (N, C, H, W)
data = data.transpose((0, 3, 1, 2))

# Save .npy files
np.save("fruit_data/train_data.npy", data)
np.save("fruit_data/train_labels.npy", labels)

print("✅ train_data.npy and train_labels.npy created successfully!")
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
