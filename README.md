# Fruit Classification Using PyTorch

This project is a simple fruit image classification system built using **PyTorch**.  
It trains a Convolutional Neural Network (CNN) to classify images of fruits into multiple categories.

---

## Project Structure

Fruit-Classification-PyTorch/
│
├─ fruit_data/ # Dataset folder (contains .npy files)
│ ├─ train_data.npy
│ ├─ train_labels.npy
│ ├─ validation_data.npy (optional)
│ └─ validation_labels.npy (optional)
│
├─ model/ # Folder where trained model is saved
├─ create_dataset.py # Optional: script to create .npy dataset files
├─ fruit_data.py # Dataset class
├─ train_model.py # Script to train the CNN
├─ test_image.py # Optional: test individual images
├─ README.md # Project instructions



---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- PIL (Python Imaging Library)
- numpy

Install the required packages with:

```bash
pip install -r requirements.txt


## Dataset

Place your dataset `.npy` files in the fruit_data/ folder:
- train_data.npy
- train_labels.npy
- validation_data.npy (optional)
- validation_labels.npy (optional)

If you don’t have .npy files, run create_dataset.py to generate them.


## Training 

python train_model.py --data-dir "fruit_data" --epochs 10

- --data-dir : Path to dataset folder
- --epochs : Number of training epochs


## Testing 

python test_image.py --image "path_to_image.jpg"


## Notes / Tips

Add extra info like:
GPU/CPU usage
  Image size requirements (64x64 RGB)
  Saved model location (model/ folder)
├─ requirements.txt # Required Python packages
└─ .gitignore
