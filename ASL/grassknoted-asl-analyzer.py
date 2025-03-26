import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, datasets

from torchmetrics.classification import MulticlassAccuracy

import kagglehub
# path = kagglehub.dataset_download("grassknoted/asl-alphabet")
# print(path)

#Creating the test dataset and dataloader
path = '/ASL/archive(3)/asl_alphabet_test/asl_alphabet_test'
path1 = path + "/space_test.jpg"
path2 = path + "/nothing_test.jpg"
if os.path.exists(path1):
    os.remove(path1)
if os.path.exists(path2):
    os.remove(path2)
transform = transforms.Compose([
    transforms.Resize((64,64)),
    # transforms.RandomRotation(30), # When rotation, flipping and normalization are applied, for some reason, all the test tensors appear to filled with 0. , with only a few exceptions
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

test_images = []
for img in os.listdir(path):
    img_name = ord(img[0]) - ord('A')
    img_path = os.path.join(path, img) 
    image = Image.open(img_path)
    img_tensor = transform(image)
    test_images.append((img_tensor, img_name))

# test_data, test_labels = zip(*test_images)

class ASLTestDataset(Dataset):
    def __init__(self, root_dir=path, transform=transform):
        self.root_dir = root_dir
        self.transform = transform
        self.images = os.listdir(root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img)
        image = Image.open(img_path)
        label = ord(img_name[0]) - ord('A')
        if self.transform:
            image = self.transform(image)
        return image, label # as a tuple (image, label)

test_data = ASLTestDataset(path, transform)
test_dataloader = DataLoader(test_data)
# for _ in range(len(test_data)):
#     print(_, test_data[_][1])

train = datasets.ImageFolder('/root/.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train', transform=transform)

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
# test_dataloader = DataLoader(test, batch_size=32, shuffle=False)

class ASLModel(nn.Module):
    def __init__(self, IN_FEATURES,HIDDEN_FEATURES,OUT_FEATURES):
        super().__init__()
        self.Conv1 = nn.Sequential( # input shape = (B,C,64,64)
            nn.Conv2d(IN_FEATURES, HIDDEN_FEATURES, kernel_size=3, stride=1, padding=1), # shape = [B,H_F,64,64]
            nn.BatchNorm2d(HIDDEN_FEATURES),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(HIDDEN_FEATURES, HIDDEN_FEATURES, kernel_size=3, stride=1, padding=1), # shape = [B,H_F,64,64]
            nn.BatchNorm2d(HIDDEN_FEATURES),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # shape = [B,H_F,32,32]
            nn.Dropout(0.5),
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(HIDDEN_FEATURES, HIDDEN_FEATURES, kernel_size=3, stride=1, padding=1), # shape = [B,H_F,32,32]
            nn.BatchNorm2d(HIDDEN_FEATURES),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(HIDDEN_FEATURES, HIDDEN_FEATURES, kernel_size=3, stride=1, padding=1), # shape = [B,H_F,32,32]
            nn.BatchNorm2d(HIDDEN_FEATURES),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # shape = [B,H_F,16,16]
            nn.Dropout(0.5),
        )

        self.analyzer = nn.Sequential(
            nn.Flatten(), # shape = [B, H_F*16*16]
            nn.Linear(HIDDEN_FEATURES*16*16, OUT_FEATURES), #shape = [B, NUM_CLASSES]
        )

    def forward(self, x:torch.tensor):
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.analyzer(x)

        return x

#HYPERPARAMETERS
NUM_CLASSES = 29
IN_FEATURES = 3
HIDDEN_FEATURES = 16
OUT_FEATURES = NUM_CLASSES
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ASLModel(IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES).to(device)

epochs = 10
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-4, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()
accuracy_fn = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

max_acc = 0
for epoch in range(epochs):
    train_loss, train_acc = 0,0
    for idx, (X_train, y_train) in enumerate(train_dataloader):
        model.train()
        X_train, y_train = X_train.to(device), y_train.to(device)
        
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        
        with torch.no_grad():
            preds = logits.squeeze().argmax(dim=1).type(torch.float)
            train_loss += loss.item()
            train_acc += accuracy_fn(y_train, preds)

        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    print(f"Train Loss : {train_loss} | Train accuracy : {train_acc}%")
    if(train_acc > max_acc):
        max_acc = train_acc
        torch.save(model.state_dict(), 'asl_cnn-model.pt')
        

test_acc = 0
for i, (X_test, y_test) in enumerate(test_dataloader):
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    preds = model(X_test).squeeze().argmax(dim=1).type(torch.float)
    test_acc += accuracy_fn(y_test, preds)

print(f"{test_acc:.3f}%")