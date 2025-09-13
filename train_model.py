import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import fruit_data
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

class FruitNet(nn.Module):
    def __init__(self):
        super(FruitNet, self).__init__()
        # Input: 3 x 64 x 64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32 x 64 x 64
        self.pool1 = nn.MaxPool2d(2, 2)                          # 32 x 32 x 32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 64 x 32 x 32
        self.pool2 = nn.MaxPool2d(2, 2)                          # 64 x 16 x 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 128 x 16 x 16
        self.pool3 = nn.MaxPool2d(2, 2)                           # 128 x 8 x 8

        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 65)  # Assuming 65 fruit classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_network(dataloader_train):
    net = FruitNet().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(epochs):
        current_loss = 0.0
        print(f"Epoch: {epoch+1}")
        for images, labels in dataloader_train:
            images, labels = images.to(device), labels.to(device)
            x = Variable(images).float()
            y = Variable(labels).long()
            optimizer.zero_grad()
            y_pred = net(x)
            correct = y_pred.max(1)[1].eq(y).sum()
            print(f"INFO: Number of correct items classified: {correct.item()}")
            loss = criterion(y_pred, y)
            print(f"Loss: {loss.item()}")
            current_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(current_loss)

    torch.save(net.state_dict(), "model/fruit_model_state_dict.pth")
    torch.save(optimizer.state_dict(), "model/fruit_model_optimizer_dict.pth")
    print(f"✅ Finished training for {epochs} epochs")
    return losses, net

def test_network(net, dataloader_test):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    accuracies = []
    with torch.no_grad():
        for feature, label in dataloader_test:
            feature, label = feature.to(device), label.to(device)
            pred = net(feature)
            accuracy = accuracy_score(label.cpu().numpy(), pred.max(1)[1].cpu().numpy()) * 100
            print("Accuracy:", accuracy)
            loss = criterion(pred, label)
            print("Loss:", loss.item())
            accuracies.append(accuracy)
    avg_acc = sum(accuracies)/len(accuracies)
    print(f"✅ Testing done with overall accuracy: {avg_acc}")

def main():
    root_dir = args.data_dir
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    transformed_dataset = fruit_data.Fruit(root_dir, train=True, transform=data_transform)
    dataloader_train = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)
    transformed_test_dataset = fruit_data.Fruit(root_dir, train=False, transform=data_transform)
    dataloader_test = DataLoader(transformed_test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Fixed iterator line
    images, labels = next(iter(dataloader_train))
    print(f"INFO: image shape: {images.shape}")
    print(f"INFO: Tensor type: {images.type()}")
    print(f"INFO: labels shape: {labels.shape}")

    losses, net = train_network(dataloader_train)
    test_network(net, dataloader_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help="Dataset directory where npy files are stored")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs")
    args = parser.parse_args()
    epochs = args.epochs
    main()
