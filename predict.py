import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import csv
import timm
import torch.nn.functional as fn


# Custom ViT without pretrained
class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        base_model = timm.create_model('vit_small_patch32_384', pretrained=False, num_classes=0)  # 不加载预训练权重
        self.base_model = base_model
        self.num_features = self.base_model.embed_dim
        # same additional layers as training model
        self.fc1 = nn.Linear(self.num_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(512, num_classes)

    # setup sequence
    def forward(self, x):
        x = self.base_model(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = fn.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = fn.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x


# Custom Dataset
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f"{i}.jpg" for i in range(30000, 40000)]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.img_names[idx]


def load_model(model_path, num_labels):
    model = CustomViT(num_classes=num_labels)
    # load weight file
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# function of prediction
def predict(model, data_loader, device, threshold=0.8):
    model.to(device)
    predictions = {}
    with torch.no_grad():
        for images, filenames in data_loader:
            images = images.to(device)
            outputs = model(images)
            sigmoid_outputs = torch.sigmoid(outputs).cpu().numpy()
            for file, probs in zip(filenames, sigmoid_outputs):
                activated_labels = np.where(probs > threshold)[0] + 1
                if activated_labels.size == 0:
                    activated_labels = np.array([np.argmax(probs) + 1])
                predictions[file] = ' '.join(map(str, activated_labels))
    return predictions

if __name__ == '__main__':
    img_dir = 'data'
    model_path = 'weights/best_model_weights.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # transform image data
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # DataLoader for test set
    test_dataset = TestDataset(img_dir=img_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=120, shuffle=False)

    # set up number of heads
    num_labels = 19
    model = load_model(model_path, num_labels)

    # get prediction with threshold above 0.6
    predictions = predict(model, test_loader, device, threshold=0.6)

    # write output prediction into prediction.csv
    with open('Predicted_labels.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['ImageID', 'Labels'])
        for img_name, labels in predictions.items():
            csv_writer.writerow([img_name, labels])

    with open('Predicted_labels.txt', 'w') as f:
        f.write("ImageID, Labels\n")
        for img_name, labels in predictions.items():
            f.write(f"{img_name}, {labels}\n")

    print("Predictions saved to Predicted_labels.csv and Predicted_labels.csv")
