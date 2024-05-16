import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
from io import StringIO
import re
from tqdm import tqdm
import timm
import torch.nn.functional as fn
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# create folder to store output weight files
Path("weights").mkdir(parents=True, exist_ok=True)


# Customise image loader of MyDataset object
def Myloader(path):
    return Image.open(path).convert('RGB')


# load dateset labels, filename, and images data
def load_data():
    FILENAME = 'train.csv'
    with open(FILENAME) as file:
        lines = [re.sub(r'([^,])"(\s*[^\n])', r'\1/"\2', line) for line in file]
    df = pd.read_csv(StringIO(''.join(lines)), escapechar="/")
    df = df.iloc[:, :2]
    labels_set = set()
    row_n = df.shape[0]
    df['Labels'].apply(lambda x: labels_set.update(map(int, x.split())))
    max_label = max(labels_set)
    n_classes = len(labels_set)
    print("all labels: ", labels_set, "n_classes: ", n_classes, "rows:", row_n)
    path = "data/%d.jpg"
    data = []
    for i in range(row_n):
        label_encode = multi_hot_encode(df['Labels'][i], max_label)
        data.append((path % i, label_encode))
    # print a sample data after processed
    print(data[-1])
    return data, max_label


# convert labels into tensor
def multi_hot_encode(label, n):
    labels = label.split()
    encode = torch.zeros(n)
    for i in labels:
        index = int(i)-1
        encode[index] += 1
    return encode


# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, data, transform=None, loader=None):
        self.data = data
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)


# Customise ViT with more layers
class CustomViT(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomViT, self).__init__()
        self.base_model = base_model
        self.num_features = self.base_model.embed_dim  # load model embedding dimension
        # new layers, heads number depends on number of classes
        self.fc1 = nn.Linear(self.num_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(512, num_classes)

    # define the sequence of the network
    def forward(self, x):
        x = self.base_model(x)  # base model layers
        # layers for novelty
        x = self.fc1(x)
        x = self.bn1(x)
        x = fn.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = fn.relu(x)
        x = self.dropout2(x)
        x = self.output(x) # final output layer
        return x


# define the process of training
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # set train mode
        model.train()
        total_train_loss = 0  # initialising train loss
        total_train_samples = 0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Training'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)
            total_train_samples += images.size(0)

        avg_train_loss = total_train_loss / total_train_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        all_predictions = []
        all_targets = []

        # loss calculations and use sigmoid activation function for output prediction
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Validation'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

                predictions = torch.sigmoid(outputs).data > 0.5  # Assume using sigmoid activation for multi-label
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}')

        # calculate last epoch validation accuracy, precision, recall, and F1 score
        if epoch == num_epochs - 1:
            val_accuracy = accuracy_score(all_targets, all_predictions)
            val_precision = precision_score(all_targets, all_predictions, average='micro')
            val_recall = recall_score(all_targets, all_predictions, average='micro')
            val_f1 = f1_score(all_targets, all_predictions, average='micro')
            print(
                f'Last Epoch: {epoch + 1}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        # save the best model with the lowest loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'weights/best_model_weights.pth')
            print(f"Saved Best Model Weights at Epoch {epoch + 1} with Val Loss: {best_val_loss:.4f}")

    # save the model of last epoch
    torch.save(model.state_dict(), 'weights/final_model_weights.pth')
    print("Saved Final Model Weights")

if __name__ == '__main__':
    # get start time for runtime experience
    start_time = datetime.datetime.now()
    data, heads_n = load_data()

    # transform image data for mitigate over fitting and resize as input layer required
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # customise Dataloader for training and validation
    dataset = MyDataset(data, transform=transform, loader=Myloader)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=450, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=450, shuffle=False, num_workers=0)

    # load pretrained vit_small_patch32_384, set num_classes=0 to remove heads and add in customised ViT
    base_model = timm.create_model('vit_small_patch32_384', pretrained=True, num_classes=0)
    model = CustomViT(base_model, num_classes=19)

    # Use combination of Sigmoid layer and the BCELoss for multiple label task loss calculation
    criterion = nn.BCEWithLogitsLoss()
    # Use AdamW as optimizer (The default weight_decay is 1e-2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # start training
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # calculate runtime
    end_time = datetime.datetime.now()
    cost_time = end_time - start_time

    print(f"start time: {start_time}")
    print(f"end time: {end_time}")
    print(f"total time cost: {cost_time}")
