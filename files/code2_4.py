# import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image

# from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from torchinfo import summary

import os
import pathlib
import shutil
import sys

import argparse
from time import sleep

class FaceCNN(nn.Module):
    def __init__(self, num_input_channels, num_classes, stride=1, padding=1):
        super().__init__()

        self.network = nn.Sequential(

            # (250, 250, 3)

            nn.Conv2d(in_channels=num_input_channels, out_channels=64, kernel_size=7, stride=2, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.2),

            nn.Flatten(),
            nn.Linear(in_features=14400, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=64),
            nn.ReLU(),       
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, input):
        output = self.network(input)
        return output

def set_seed(seed=0):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

def evaluate(loader, model):
    model.eval()

    score = 0
    cnt = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            score += float(torch.sum(pred==labels.data))
            cnt += pred.shape[0]

    return score/cnt

def train():
    print("-"*15+"Training"+"-"*15)
    best_acc = 0.0

    epoch = 0
    
    while epoch<num_epochs:

        train_score = 0
        cnt = 0
        train_loss = 0

        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # print(outputs, labels)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            train_score += float(torch.sum(preds==labels.data))
            cnt += inputs.shape[0]

            # print(preds, labels)

        train_acc = train_score/cnt
        val_acc = evaluate(val_loader, model)
        
        print("Epoch:", epoch, "\tLoss:", train_loss, "\tTraining Acc:", train_acc, "\tVal Acc:", val_acc)
        
        epoch += 1

        if val_acc > best_acc:
            torch.save(model.state_dict(),'best.model')
            best_acc = val_acc


def train_n_epochs(num_curr_epochs, path_saved, process_id, used_mem):
    print("-"*15+"Training_"+str(num_curr_epochs)+"-"*15)

    if os.path.exists(path_saved):
        checkpoint = torch.load(path_saved)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
    else:
        epoch = 0
        best_acc = 0.0
    
    curr_run_epoch = 0

    while curr_run_epoch<num_curr_epochs:

        train_score = 0
        cnt = 0
        train_loss = 0

        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # print(outputs, labels)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            _, preds = torch.max(outputs.data, 1)
            train_score += float(torch.sum(preds==labels.data))
            cnt += inputs.shape[0]

            # print(preds, labels)

        train_acc = train_score/cnt
        val_acc = evaluate(val_loader, model)
        
        print("Epoch:", epoch, "\tLoss:", train_loss, "\tTraining Acc:", train_acc, "\tVal Acc:", val_acc)
        
        curr_run_epoch += 1
        epoch += 1

        if val_acc > best_acc:
            # torch.save(model.state_dict(),f'best_{process_id}.model')     # save the best model till now
            best_acc = val_acc
        
        mem_tuple = torch.cuda.mem_get_info(device=torch.cuda.current_device())
        used_mem[0] = max(used_mem[0], mem_tuple[1]-mem_tuple[0])

    # torch.save(model, 'model_'+str(os.getpid())+'.pt')        # for inference
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        }, path_saved)


def test(path_saved):
    print("-"*15+"Testing"+"-"*15)

    if os.path.exists(path_saved):
        checkpoint = torch.load(path_saved)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Model not found at path", path_saved)
        return

    model.eval()
    score = 0
    cnt = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=non_blocking), labels.to(device, non_blocking=non_blocking)
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            score += float(torch.sum(pred==labels.data))
            cnt += pred.shape[0]

    print("Test accuracy:", score/cnt)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int)
    parser.add_argument('-i', '--id', help='Id', type=int)
    parser.add_argument('-t', '--test', help='Test', type=int)
    parser.add_argument('-n', '--number', help='Total number of processes. Used for computing fraction of GPU memory.', type=int)
    args = parser.parse_args()

    set_seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    # device = torch.device("cpu")

    print("device:", device)

    # if args.number is not None and device==torch.device("cuda"):
    #     torch.cuda.set_per_process_memory_fraction(0.5/args.number, device=torch.cuda.current_device())

    used_mem = [0]      # using list, to "pass by reference"

    data_folder = './data/'

    num_ppl = 4

    data_root = './data/'
    dirs = ['train', 'val', 'test']

    train_path = os.path.join(data_root, dirs[0])
    val_path = os.path.join(data_root, dirs[1])
    test_path = os.path.join(data_root, dirs[2])

    train_transform = transforms.Compose(transforms=[
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose(transforms=[
        transforms.ToTensor(),
    ])

    dataloader_kwargs = {
        'pin_memory': False,
        'num_workers': 1,
        'batch_size': 2,
        'shuffle': True
    }
    non_blocking = dataloader_kwargs['pin_memory']

    train_loader = DataLoader(
        torchvision.datasets.ImageFolder(train_path, train_transform), **dataloader_kwargs
    )
    val_loader = DataLoader(
        torchvision.datasets.ImageFolder(val_path, test_transform), **dataloader_kwargs
    )
    test_loader = DataLoader(
        torchvision.datasets.ImageFolder(test_path, test_transform), **dataloader_kwargs
    )

    num_input_channels = 3
    model = FaceCNN(num_input_channels=num_input_channels, num_classes=num_ppl).to(device)
    # print(summary(model, input_size=(dataloader_kwargs['batch_size'], num_input_channels, 250, 250)))

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 3

    # path_saved = 'model_'+str(os.getpid())+'.pt'

    process_id = args.id if args.id is not None else args.test
    path_saved = str(process_id)+'.pt'

    if args.test is not None:
        test(path_saved)
    else:
        train_n_epochs(args.epochs, path_saved, process_id, used_mem)

    # print(torch.cuda.max_memory_allocated(device=torch.cuda.current_device()))
    # print(torch.cuda.memory_summary(device=torch.cuda.current_device()))

    print("Peak mem usage:", used_mem[0])