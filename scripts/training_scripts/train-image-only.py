#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Title: train-image-only.py
#
# Description: This script trains a ResNet18 model to predict fish ages using
# fish scale images only. Arguments, hyperparameters, and other settings are
# included in a configs.yml file. The trained model is saved to a specified
# output directory.
#
# Author: aotian.zheng@noaa.gov
# Release Date: July 2025
# Last Updated: September 2025
#
# Usage: python train-image-only.py -c path/to/configurations.yml
# -----------------------------------------------------------------------------

import argparse
import yaml
import copy
import cv2 as cv
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

class FishTestDataset(Dataset):
    """Custom Dataset for loading fish scale images for age inference.
    
    Attributes
    ----------
    image_dir : str
        Path to the directory containing images.
    image_name : list
        List of image filenames in the directory.
    transforms : callable, optional
        A function/transform that takes in a PIL image and returns a transformed version.
    
    Methods
    -------
    __len__ : returns the number of images in the dataset.
    __getitem__(index) : returns the image and its filename at the specified index.
    """
    def __init__(self, image_dir, csv_path, transform=None):
        """
        Parameters
        ----------
        image_dir : str
            Path to the directory containing images.
        csv_path : str
            Path to the CSV file containing metadata and known age labels.
        transform : callable, optional
            A function/transform that takes in a PIL image and returns a transformed version.
        """

        # Read the metadata csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        
        # Get the directory dataset images
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform

        # Image name
        self.image_name = np.asarray(self.data_info.iloc[:, 0])
        
        # Extract metadata attributes: fish length, weight, month of catch, known age
        self.length = np.asarray(self.data_info.iloc[:, 1])
        self.wt = np.asarray(self.data_info.iloc[:, 2])
        self.month = np.asarray(self.data_info.iloc[:, 3])
        self.age = np.asarray(self.data_info.iloc[:, 4])

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_name)

    def __getitem__(self, index):
        """Returns the image and its filename at the specified index."""
        # Open the specified image
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)
        
        # Normalize metadata
        wt_l_m = torch.tensor([(self.wt[index] - 163)/(82), (self.length[index] - 211)/ (35.5), (self.month[index]-7.4)/(1.9)]).type(torch.FloatTensor)

        # Replaces ages greater than 4 with 4 to create 5 age classes (0, 1, 2, 3, 4+)
        if(self.age[index] < 5):
          label_age = self.age[index]
        else:
          label_age = 4
            
        # Transform the image, if transforms are provided
        if self.transforms:
            image = self.transforms(image)

        return (image,wt_l_m) , self.image_name[index], label_age
        
def main():
    """Main function to train the ResNet18 model using fish scale images."""
    # Parse command line arguments. Currently only requires a path to a configuration yaml file.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="path to configuration yaml file")
    args = parser.parse_args()

    # Open the configuration file and read in the parameters
    try:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The configuration file was not found at {args.config_path}")
        return

    # Check for file names included in config paths where needed
    if ".csv" not in config["train_csv"]:
        raise ValueError("The 'train_csv' key in the configuration file must include a file name ending with '.csv'.")
    if ".csv" not in config["validation_csv"]:
        raise ValueError("The 'validation_csv' key in the configuration file must include a file name ending with '.csv'.")

    # Image transformations: resizing, cropping, normalization
    data_dir = config["train_img_path"]
    data_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    # Load and shuffle the training and validation datasets
    train_dataset = FishTestDataset( data_dir, config["train_csv"], data_transforms)
    val_dataset = FishTestDataset( data_dir, config["validation_csv"], data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, drop_last=False)

    # Use GPU, if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the ResNet model
    # Number of classes corresponds to the number of age classes.
    model = resnet18(num_classes = 5).to(device)

    # Load pretrained model weights
    loaded_state_dict = torch.hub.load_state_dict_from_url("https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth")
    current_model_dict = model.state_dict()
    new_state_dict={k:v if v.size()==current_model_dict[k].size()  else  current_model_dict[k] for k,v in zip(current_model_dict.keys(), loaded_state_dict.values())}

    # Model hyperparameters
    num_epochs = config["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config["scheduler"], gamma=config["gamma"])
    criterion = nn.CrossEntropyLoss()

    # Create the model_out_path directory if it doesn't exist
    if(not os.path.exists(config["model_out_path"])):
        os.mkdir(config["model_out_path"])

    # Keep track of model accuracy during training
    best_acc = 0

    # Train the model
    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        running_res = []

        # Keep track of stats
        running_loss = 0.0
        running_corrects = 0
        running_corr = [0.0, 0.0, 0.0, 0.0, 0.0]
        running_total = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Loop through each image
        for images, imagename, labels in tqdm(train_loader, desc="Training model"):
            # Send example to GPU
            images = images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = model(images)#inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
            # Predict ages (returns certainty for each age class)
            _, preds = torch.max(output, 1)

            # Loss and accuracy
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            #Loop through each age class
            for i in range(0, len(preds)):
                if labels.data[i].cpu().detach().numpy() == 3:
                    count_3 += 1

                if preds[i] == labels.data[i]:
                    running_corr[int(labels.data[i].cpu().detach().numpy())] += 1.0
                running_total[int(labels.data[i].cpu().detach().numpy())] += 1.0

        # Training loss and accuracy
        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * running_corrects / len(train_loader.dataset)
        running_res = [100.0 * i / max(1,j) for i, j in zip(running_corr, running_total)]
        print(running_res)
        print("{} Loss: {:.4f} Average Accuracy: {:.4f}".format("train", epoch_loss, epoch_acc))

        # VALIDATION
        model.eval()
        running_res = []

        # Keep track of stats
        running_loss = 0.0
        running_corrects = 0
        running_corr = [0.0, 0.0, 0.0, 0.0, 0.0]
        running_total = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Loop through each image
        for images, imagename, labels in tqdm(val_loader, desc="Validating current model"):
            # Send example to GPU
            images = images.to(device)
            labels = labels.to(device)
        
            # Zero the parameter gradients
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                output = model(images)#inputs)
                
            # Age prediction (returns certainty for each age class)
            _, preds = torch.max(output, 1)

            # Loss and accuracy
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # Loop through each age class
            for i in range(0, len(preds)):
                if labels.data[i].cpu().detach().numpy() == 3:
                    count_3 += 1

                if preds[i] == labels.data[i]:
                    running_corr[int(labels.data[i].cpu().detach().numpy())] += 1.0
                running_total[int(labels.data[i].cpu().detach().numpy())] += 1.0

        # Validation loss and accuracy
        scheduler.step()
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = 100.0 * running_corrects / len(val_loader.dataset)
        running_res = [100.0 * i / max(1,j) for i, j in zip(running_corr, running_total)]
        print(running_res)
        print("{} Loss: {:.4f} Average Accuracy: {:.4f}".format("validation", epoch_loss, epoch_acc))

        # Save the best model weights
        if(epoch_acc > best_acc):
            print("saving best model")
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            res = running_res.copy()
            torch.save(model.state_dict(), config["model_out_path"]+'/best_model.pth')

    # Save the best model to file
    torch.save(model.state_dict(), config["model_out_path"]+'/final_model.pth')

if __name__ == "__main__":
    main()