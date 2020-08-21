# -*- coding: utf-8 -*-
"""
This Script contains the CNN dropout with Spinal fully-connected layer.


@author: Dipu
"""
import shutil

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import random

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import os
import torch.nn.functional as F
import numpy as np
import cv2
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 15
learning_rate = 0.0001

torch.manual_seed(1)
random.seed(1)

Half_width = 2048
layer_width = 128

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(256, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
])

class CustomData(Dataset):
    def __init__(self, file_list, dir, transform=None):
        self.file_list = file_list
        self.dir = dir
        self.transform = transform

        if 'court00' in self.file_list[0]:
            self.label = 0
        elif 'court11' in self.file_list[0]:
            self.label = 1
        elif 'court22' in self.file_list[0]:
            self.label = 2
        elif 'court33' in self.file_list[0]:
            self.label = 3
        elif 'tax00' in self.file_list[0]:
            self.label = 4
        elif 'tax11' in self.file_list[0]:
            self.label = 5
        else:
            self.label = 6

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(
            (os.path.join(self.dir, self.file_list[idx])))
        if self.transform:
            img = self.transform(img)
        img = img.numpy()
        return img.astype('float32'), self.label



train_dir = 'C:\\Users\\Admin\\Desktop\\etc\\court(naming)\\train_all'
test_dir = 'C:\\Users\\Admin\\Desktop\\etc\\court(naming)\\test_all'
train_files = os.listdir(train_dir)
test_files = os.listdir(test_dir)
court_files_00 = [tf for tf in train_files if 'court00' in tf]
court_files_11 = [tf for tf in train_files if 'court11' in tf]
court_files_22 = [tf for tf in train_files if 'court22' in tf]
court_files_33 = [tf for tf in train_files if 'court33' in tf]
tax_files_00 = [tf for tf in train_files if 'tax00' in tf]
tax_files_11 = [tf for tf in train_files if 'tax11' in tf]
tax_files_22 = [tf for tf in train_files if 'tax22' in tf]
test_court_files_00 = [tf for tf in test_files if 'court00' in tf]
test_court_files_11 = [tf for tf in test_files if 'court11' in tf]
test_court_files_22 = [tf for tf in test_files if 'court22' in tf]
test_court_files_33 = [tf for tf in test_files if 'court33' in tf]
test_tax_files_00 = [tf for tf in test_files if 'tax00' in tf]
test_tax_files_11 = [tf for tf in test_files if 'tax11' in tf]
test_tax_files_22 = [tf for tf in test_files if 'tax22' in tf]

train_00 = CustomData(court_files_00, train_dir, transform=transform_train)
train_11 = CustomData(court_files_11, train_dir, transform=transform_train)
train_22 = CustomData(court_files_22, train_dir, transform=transform_train)
train_33 = CustomData(court_files_33, train_dir, transform=transform_train)
train_44 = CustomData(tax_files_00, train_dir, transform=transform_train)
train_55 = CustomData(tax_files_11, train_dir, transform=transform_train)
train_66 = CustomData(tax_files_22, train_dir, transform=transform_train)
test_00 = CustomData(test_court_files_00, test_dir, transform=transform_test)
test_11 = CustomData(test_court_files_11, test_dir, transform=transform_test)
test_22 = CustomData(test_court_files_22, test_dir, transform=transform_test)
test_33 = CustomData(test_court_files_33, test_dir, transform=transform_test)
test_44 = CustomData(test_tax_files_00, test_dir, transform=transform_test)
test_55 = CustomData(test_tax_files_11, test_dir, transform=transform_test)
test_66 = CustomData(test_tax_files_22, test_dir, transform=transform_test)

_dataset = ConcatDataset([train_00, train_11, train_22, train_33, train_44, train_55, train_66])
test_dataset = ConcatDataset([test_00, test_11, test_22, test_33, test_44, test_55, test_66])

#1 train = 162 , test = 50
train_loader = DataLoader(_dataset, batch_size=9, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)




# 3x3 convolution
class SpinalCNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(SpinalCNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(Half_width + layer_width, layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.1), nn.Linear(layer_width * 4, 7)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, Half_width:2 * Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, Half_width:2 * Half_width], x3], dim=1))

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)

        return x


class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.fc_layer = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(512, 3)
        # )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            # nn.Linear(65536,16384),
            # nn.ReLU(inplace=True),
            nn.Linear(16384, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 7)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        return x


model = SpinalCNN().to(device)
# model = CNN().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#Train the model
total_step = len(train_loader)
curr_lr = learning_rate
best_accuracy = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 9 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        if best_accuracy> correct / total:
            curr_lr = curr_lr/3
            update_lr(optimizer, curr_lr)

        model.train()
file_path = r'C:\Users\Admin\Desktop\Model\sample_model\model_parameter(test).pth'

torch.save(model.state_dict(), file_path)

print("finish save")

def test():
    # for images, labels in train_loader:
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
    classes = ('00', '11', '22', '33', '44', '55', '66')
    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()
    a = labels.size()
    b = labels

    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(7)))

    net = model
    net.load_state_dict(torch.load(
        r'C:\Users\Admin\Desktop\Model\model_parameter1(epoch50).pth'))
    net.eval()
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(7)))

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
        # for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %d %%' % (
            100 * correct / total))

    class_correct = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    with torch.no_grad():
        for images, labels in test_loader:
        # for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)


            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(10):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(7):
        if (class_total[i] != 0):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))



def singleTest(Path):
    image1 = Image.open(Path)
    input1 = transform_test(image1)
    input1 = input1.to(device)
    net1 = model
    net1.load_state_dict \
            (torch.load(
            r'C:\Users\Admin\PycharmProjects\ResNet1\All\model_parameter(test).pth'))
    net1.eval()

    input1 = input1.view(1, 3, 256, 256)

    output = net1(input1)

    _, prediction = torch.max(output, 1)
    if torch.max(output).item() > 10:
        if prediction.item() == 0:
            print('Predicted : 00 label')
            shutil.copy(Path, r'C:\Users\Admin\Desktop\etc\taxResult\00')
        if prediction.item() == 1:
            print('Predicted : 11 label')
            shutil.copy(Path, r'C:\Users\Admin\Desktop\etc\taxResult\11')
        if prediction.item() == 2:
            print('Predicted : 22 label')
            shutil.copy(Path, r'C:\Users\Admin\Desktop\etc\taxResult\22')
    else:
        print('Prediction err')
        shutil.copy(Path, r'C:\Users\Admin\Desktop\etc\taxResult\etc')


def MultiTest(Path):
    file_list = os.listdir(Path)
    for idx in range(len(file_list)):

        image1 = Image.open(Path + '/' + file_list[idx])
        input1 = transform_test(image1)
        input1 = input1.to(device)
        net1 = model
        net1.load_state_dict \
                (torch.load(
                r'C:\Users\Admin\PycharmProjects\ResNet1\All\model_parameter(test).pth'))
        net1.eval()
        input1 = input1.view(1, 3, 256, 256)
        output = net1(input1)
        x = output.tolist()
        z = torch.tensor(x)
        sfmax = F.softmax(z, dim=1)
        scores = torch.max(sfmax).item()
        _, prediction = torch.max(output, 1)
        Path1 = Path + '/' + file_list[idx]
        sc = torch.max(output).item()
        filename = file_list[idx]
        # filename = filename[:-4] + "_{:0.5f}.jpg".format(scores)
        filename = "_{:0.5f}".format(scores) + filename
        numimg = np.asarray(image1)
        print(scores)
        # if scores > 0.8:
        if prediction.item() == 0:
            print('Predicted : 00 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/00/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\00')
        elif prediction.item() == 1:
            print('Predicted : 11 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/11/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\11')
        elif prediction.item() == 2:
            print('Predicted : 22 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/22/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\22')
        elif prediction.item() == 3:
            print('Predicted : 33 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/33/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\22')
        elif prediction.item() == 4:
            print('Predicted : 44 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/44/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\22')
        elif prediction.item() == 5:
            print('Predicted : 55 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/55/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\22')
        elif prediction.item() == 6:
            print('Predicted : 66 label')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/66/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\22')

        else:
            print('Predicted : etc label===============================')
            filepathname = 'C:/Users/Admin/Desktop/etc/AllResult/etc/' + filename
            cv2.imwrite(filepathname, numimg)
            # shutil.copy(Path1, r'C:\Users\Admin\Desktop\etc\taxResult\22')



if __name__ == "__main__":
#     test()
    # singleTest(r'C:\Users\Admin\Desktop\etc\court(naming)\test_tax\tax00_01.jpg')
    MultiTest(r'C:\Users\Admin\Desktop\etc\1')