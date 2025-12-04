#!/usr/bin/env python

# -----------------------------------------------------------------------------
# Title: predict-ages-images.py
#
# Description: This script predicts the age of fish scale images using a pre-
#              trained ResNet18 model. Arguments, hyperparameters, and other
#              settings are included in a configs.yml file. Predicted ages are
#              written to a CSV file.
#
# Author: aotian.zheng@noaa.gov
# Release Date: July 2025
# Last Updated: September 2025
#
# Usage: python predict-ages-images.py -c path/to/configurations.yml
# -----------------------------------------------------------------------------

import argparse
import warnings
import yaml
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

# Load configuration file, reluctantly handling Windows directories
def load_yaml(file):
    """Load YAML file `file` while reluctantly handling Windows directory
    backslashes. Tries to load the file normally first. If this fails, the file
    is read in as a text string, any offending characters are replaced, and the
    string is then converted to YAML. In this case, a warning advising safer
    syntax is thrown."""
    # Try to load YAML normally
    try:
        with open(file, 'r') as f:
            config = yaml.safe_load(f)
    # Catch, warn about, and handle invalid escape characters that prevent
    # normal loading
    except yaml.YAMLError:
        warnings.warn("One or more file paths in the configuration file contain invalid escape characters. To fix this, enclose directory paths with single quotations ('...') or use all forward slashes (/) or double backslashes (\\\\) in directory paths. We will force-read as-is, but beware that unexpected bad things may happen.", SyntaxWarning)
        with open(file, 'r') as f:
            temp = f.read()
        temp = temp.replace('\\', '/')
        config = yaml.safe_load(temp)
    return config

def fix_config(config):
    """Fix some common problems that may arise with configuration entries"""
    for k, v in config.items():
        if '_path' in k:
            # Fix capitalized file extensions
            _, ext = os.path.splitext(v)
            v = v.replace(ext.upper(), ext.lower())
            # Platform-agnostic directory paths
            config[k] = os.path.join(v)
    return config

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
    def __init__(self, image_dir, transform=None):
        """
        Parameters
        ----------
        image_dir : str
            Path to the directory containing images.
        transform : callable, optional
            A function/transform that takes in a PIL image and returns a transformed version.
        """

        # Get the directory of the images to age
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform

        # Image Name
        self.image_name = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_name)

    def __getitem__(self, index):
        """Returns the image and its filename at the specified index."""
        # Open the specified image
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)
        
        # Transform the image, if transforms are provided
        if self.transforms:
            image = self.transforms(image)

        return image, self.image_name[index]

def main():
    """Main function to execute the inference script.""" 
    # Parse command line arguments. Currently only requires a path to a configuration yaml file.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="path to configuration yaml file")
    args = parser.parse_args()
    

    # Open the configuration file and read in the parameters
    try:
        config = load_yaml(file=args.config_path)
        config = fix_config(config)
    except FileNotFoundError:
        print(f"Error: The configuration file was not found at {args.config_path}")
        return
        
    # Format directories for cross-platform compatibility
    config.update({k: os.path.join(i) for k,i in config.items if 'path' in k})

    # Check for file names included in config paths where needed
    if ".csv" not in config["metadata_path"]:
        raise ValueError("The 'metadata_path' key in the configuration file must include a file name ending with '.csv'.")
    if ".csv" not in config["out_path"]:
        raise ValueError("The 'out_path' key in the configuration file must include a file name ending with '.csv'.")
    if ".pth" not in config["model_path"]:
        raise ValueError("The 'model_path' key in the configuration file must include a file name ending with '.pth'.")

    # Image transformations: resizing, cropping, normalization
    data_transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    test_dataset = FishTestDataset(image_dir=config["image_path"], transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

    # Load the model using GPU, if available, in evaluation mode.
    # Number of classes corresponds to the number of age classes.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(num_classes=5)

    # Load the pre-trained model weights
    try:
        model.load_state_dict(torch.load(config["model_path"]))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
    model.eval()    
    model.to(device)

    # Create output file and write header
    try:
        with open(config["out_path"], 'w') as file:
            file.write("Image Name, Predicted Age\n")

            # Loop through the dataset and make predictions
            for images, img_path in tqdm(test_loader, desc="Predicting ages"):
                images = images.to(device)
                outputs = model(images)
                outputs = torch.squeeze(outputs)
                _, preds = torch.max(outputs, 1)
                preds = preds.cpu().detach().numpy()
                for i in range(preds.shape[0]):
                    age = str(preds[i])
                    # Change the maximum age class to "4+"
                    if(preds[i] == 4):
                        age = "4+"
                    file.write("%s,%s\n" % (img_path[i], age))
        print(f"Inference complete. Results saved to {config['out_path']}")
    
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()