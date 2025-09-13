import numpy as np
import os

os.makedirs("fruit_data", exist_ok=True)

# Create fake training data: 10 images, 3 channels, 64x64
train_data = np.random.randint(0, 256, size=(10, 3, 64, 64), dtype=np.uint8)
train_labels = np.random.randint(0, 65, size=(10,), dtype=np.int64)  # 65 classes

# Create fake validation data: 3 images
val_data = np.random.randint(0, 256, size=(3, 3, 64, 64), dtype=np.uint8)
val_labels = np.random.randint(0, 65, size=(3,), dtype=np.int64)

# Save files
np.save("fruit_data/train_data.npy", train_data)
np.save("fruit_data/train_labels.npy", train_labels)
np.save("fruit_data/validation_data.npy", val_data)
np.save("fruit_data/validation_labels.npy", val_labels)

print("✅ Fake dataset created successfully!")
