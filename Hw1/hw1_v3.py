import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import time
import os
import PIL.Image as Image
from IPython.display import display
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

train_path = 'training_data/training_data/'
test_path = 'testing_data/testing_data/'
train_label_path = 'training_labels.csv'


def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(Dataset):
    def __init__(self, path, label, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        class_names = []
        row_number = 0
        

        df = pd.read_csv(
            "C:/Users/daniel/Desktop/CS_T0828_HW1/training_labels.csv")
        pd.set_option("display.max_rows", None)
        class_names = df["label"].value_counts().index.tolist()
        # print(class_names)
        for row in df.iterrows():
            # print(str(row[1][0]).zfill(6))
            imgs.append((str(row[1][0]).zfill(6),row[1][1]))
            
        self.path = path
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        label_idx = self.classes.index(label)
        img = self.loader(os.path.join(self.path, str(fn) + '.jpg'))
        if self.transform is not None:
            img = self.transform(img)
        return img, label_idx

    def __len__(self):
        return len(self.imgs)

def train_model(model, criterion, optimizer, scheduler, n_epochs = 5):
    
    losses = []
    accuracies = []
    test_accuracies = []
    # set the model to train mode initially
    model.train()
    for epoch in range(n_epochs):
        since = time.time()
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(trainloader, 0):

            # get the inputs and assign them to cuda
            inputs, labels = data[0], data[1]
            #inputs = inputs.to(device).half() # uncomment for half precision model
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # calculate the loss/acc later
            running_loss += loss.item()
            running_correct += (labels==predicted).sum().item()

        epoch_duration = time.time()-since
        epoch_loss = running_loss/len(trainloader)
        epoch_acc = 100/32*running_correct/len(trainloader)
        print("Epoch %s, duration: %d s, loss: %.4f, acc: %.4f" % (epoch+1, epoch_duration, epoch_loss, epoch_acc))
        
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_acc = eval_model(model)
        test_accuracies.append(test_acc)
        
        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
        since = time.time()
    print('Finished Training')
    return model

def eval_model(model):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0], data[1]
            #images = images.to(device).half() # uncomment for half precision model
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %d %%' % (
        test_acc))
    return test_acc

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(device))

    df = pd.read_csv(
        "C:/Users/daniel/Desktop/CS_T0828_HW1/training_labels.csv")
    pd.set_option("display.max_rows", None)
    table = df["label"].value_counts()

    train_tfms = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_arg = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    dataset = myImageFloder(
        path=train_path, label='training_labels.csv', transform=train_tfms)
    dataset_arg = myImageFloder(
        path=train_path, label='training_labels.csv', transform=train_arg)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    train_dataset = torch.utils.data.ConcatDataset(
        [train_dataset, dataset_arg])

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=24, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=24, shuffle=True, num_workers=4)

    model_ft = models.resnet50(pretrained=True)
    # model_ft = net.Net()
    num_ftrs = model_ft.fc.in_features

    # model_ft.fc = nn.Linear(num_ftrs, 196)
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 5000),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(5000, 500),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(500, 196)
    )
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)

  
    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, threshold=0.9)

    model_ft = train_model(model_ft, criterion, optimizer,
                           lrscheduler, n_epochs=40)
    # torch.save(model_ft.state_dict(), 'resnet50.pth')

    # pre_dic = torch.load('resnet50.pth')
    # model_ft.load_state_dict(pre_dic)

    classes = dataset.classes
    result = []  # predictions result
    print('testing')
    allFileList = os.listdir(test_path)
    with torch.no_grad():
        for file in allFileList:
            img = Image.open(test_path + file).convert('RGB')
            img = train_tfms(img).unsqueeze(0)
            img = img.to(device)
            output = model_ft(img)
            # get the label with highest value
            _, predicted = torch.max(output, 1)
            result.append([file.split('.')[0], classes[predicted.item()]])

    print('generate predictions.csv')
    with open('predictions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id', 'label'])
        writer.writerows(result)
    print('Finish')
