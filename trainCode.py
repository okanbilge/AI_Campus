import torch
import numpy as np
from pathlib import Path
import cv2
import math
from sklearn.metrics import recall_score,precision_score, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from torchvision import transforms, models
import tqdm
import csv
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models import *
from transformations import *
from sklearn.model_selection import KFold, train_test_split
import random

device="cuda:0"

class StandardScaler():
    def __init__(self) -> None:
        self.mean=None
        self.std=None
    def fit(self,tensor:torch.Tensor) -> None:
        self.mean=tensor.mean((0,2,3),keepdim=True)
        self.std=tensor.std((0,2,3),keepdim=True)
    def transform(self,tensor:torch.Tensor) -> torch.Tensor:
        scaled=(tensor-self.mean)/(self.std+1e-5)
        return scaled
    def fit_transform(self,tensor:torch.Tensor) -> torch.Tensor:
        self.fit(tensor=tensor)
        scaled=self.transform(tensor=tensor)
        return scaled

class MyTrainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, test_loader, log_dir,model_name, patience=5):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.best_model_state_dict = None
        self.best_val_loss = float('inf')
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.patience = patience
        self.model_name = model_name
        self.early_stop_counter = 0

    def train(self, epochs):
        for epoch in tqdm.tqdm(range(epochs)):
            print(f"Epoch: {epoch+1}")
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state_dict = self.model.state_dict()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            print(f"Best validation loss: {self.best_val_loss:.4f}")

            if self.early_stop_counter >= self.patience:
                print("Early stopping triggered")
                break
        if self.best_model_state_dict is not None:
            self.model.load_state_dict(self.best_model_state_dict)
            torch.save(self.model.state_dict(), "./ModelsV2/"+self.model_name+"_best_model.pth")
            print("Model saved to best_model.pth")

    def train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs.to(device))
            loss = self.loss_fn(outputs, labels.to(device))
            loss.backward()
            self.optimizer.step()

            # Log training loss to TensorBoard
            self.writer.add_scalar('Training Loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

    def validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data in self.val_loader:
                inputs, labels = data
                outputs = self.model(inputs.to(device))
                val_loss += self.loss_fn(outputs, labels.to(device)).item()
            val_loss /= len(self.val_loader)

        # Log validation loss to TensorBoard
        self.writer.add_scalar('Validation Loss', val_loss, epoch)
        
        return val_loss

    def test(self,test_loader,modelName):
        self.model.eval()
        labels = []
        test_preds = []
        with torch.no_grad():
            for x, y in self.test_loader:
                pred = self.model(x.to(device)).squeeze(-1)
                labels.extend(y.cpu().numpy())
                test_preds.extend(pred.cpu().numpy())
        test_preds = np.asarray(test_preds)
        labels_1 = np.asarray(labels)

        test_preds_1 = np.argmax(test_preds, axis=1)

        print(classification_report(labels_1, test_preds_1))
        cm_test = confusion_matrix(labels_1, test_preds_1)
        t1 = ConfusionMatrixDisplay(cm_test)

        t1.plot()
        fig = plt.gcf()
        plotPath = './Plots/' + str(modelName)  + '.png'
        fig.savefig(plotPath)
        plt.close(fig)

        precision = precision_score(labels_1, test_preds_1)
        recall = recall_score(labels_1, test_preds_1)
        accuracy = accuracy_score(labels_1, test_preds_1)
        print(cm_test.shape)
        newLine = [modelName, accuracy, recall, precision, plotPath, np.reshape(cm_test,(-1))]
        print(newLine)
        with open('./Results-TB.csv', 'a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(newLine)

        # Log test metrics to TensorBoard
        self.writer.add_scalar('Test Accuracy', accuracy)
        self.writer.add_scalar('Test Precision', precision)
        self.writer.add_scalar('Test Recall', recall)

        # Return test results
        return precision, recall, accuracy




class TBChestInMemoryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: 1 if i ==0 else 0 for i in range(len(self.classes))}
        self.images = self._load_images()
        self.transform = transform

    def _load_images(self):
        images = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path).convert('RGB')
                images.append((img, self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

root_dir = "./Datasets/TB_Chest_Radiography_Database/"

train = 1
random.seed(42)
device = "cuda:0"
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

transformType = 4

if transformType == 1:
    transform = transform_v1
elif transformType == 2:
    transform = transform_v2
else :
    transform = transform_v3

random.seed(42)

dataset = TBChestInMemoryDataset(root_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for fold, (train_index, test_index) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold + 1}:")
    train_indices, val_indices = train_test_split(train_index, test_size=0.2, shuffle=True, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    print(train_indices[:10])
    if train ==1:
        for modelID in range(9):

            model = modelList[modelID].to(device)
            
            loss_fn=torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


            modelNames = ['alexnet', 'vgg16', 'resnet50', 'densenet121', 'shufflenet_v2_x1_0', 'resnext50_32x4d', 'wide_resnet50_2',"vit_base_patch16_224", "vit_base_patch32_224"]
            modelName = modelNames[modelID] + "-"+ str(fold+1)+ "-"+ str(transformType)
            print(modelName)
            log_dir = "./logs/"+ str(modelName) + "-"+ str(fold+1) + "-" + str(transformType) 
            trainer = MyTrainer(model, loss_fn, optimizer, train_loader, val_loader, test_loader, log_dir,modelName)
            trainer.train(epochs=50)
            trainer.test(test_loader,modelName)
