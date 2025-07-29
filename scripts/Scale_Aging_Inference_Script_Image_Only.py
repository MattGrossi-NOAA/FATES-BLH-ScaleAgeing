
# -----
# PyTorch machine learning tools
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights


import argparse
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

def crop_and_pad(image, kernel_size = 10, threshold=100, pad=0.05, bottom_pad = 0.35):
    """
    Crops and pads an input image based on binary thresholding and contour detection.
    Args:
        image: Input image (numpy array).
        kernel_size: Size of kernel for morphological operations.
        threshold: Threshold value for binarization.
        pad: Fractional padding for left/right/top.
        bottom_pad: Fractional padding for bottom.
    Returns:
        Cropped and padded image (numpy array).
    """
    #Binary threshold
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_image,threshold,255,cv.THRESH_BINARY)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    
    #Morphological operation
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)
    
    #Find Contours
    contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if(len(contours) > 0):
        maxidx = 0
        maxarea = 0
        for i in range(len(contours)):
            contour = contours[i]
            if cv.contourArea(contour) > maxarea:
                maxarea = cv.contourArea(contour)
                maxidx = i
        x,y,w,h = cv.boundingRect(contours[maxidx])
        
    #Crop and Pad
    xmargin = pad
    ymargin_top = pad
    ymargin_bot = bottom_pad
    y_max, x_max,c = image.shape
    x_new = int(max(0,x-w*xmargin))
    y_new = int(max(0,y-h*ymargin_top))
    w_new = int(min(w*(1+xmargin*2),x_max-x_new))
    h_new = int(min(h*(1+ymargin_top+ymargin_bot), y_max-y_new))
    crop = image[y_new:y_new+h_new,x_new:x_new+w_new,:]
    
    c_h, c_w, c = crop.shape
    left, right, top, bottom = 0,0,0,0
    if(c_h>c_w):
        difference = c_h-c_w
        left = int(difference/2)
        right = int(difference/2)
    if(c_h<c_w):
        difference = c_w-c_h
        top = int(difference/2)
        bottom = int(difference/2)
    crop_pad = cv.copyMakeBorder(crop, top, bottom, left, right, cv.BORDER_REPLICATE)

    return crop

def preprocess_folder(image_dir, output_dir, extension = ".tif", kernel_size = 10, threshold=100, pad=0.05, bottom_pad = 0.35):
    """
    Processes all images in a folder: crops and pads them, then saves as .jpg.
    Args:
        image_dir: Directory containing raw images.
        output_dir: Directory to save processed images.
        extension: File extension to filter images.
        kernel_size, threshold, pad, bottom_pad: Parameters for crop_and_pad.
    """
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    for file in os.listdir(image_dir):
        if file.endswith(extension):
            image = cv.imread(os.path.join(image_dir, file))
            cropped_image = crop_and_pad(image, kernel_size, threshold, pad, bottom_pad)
            cv.imwrite(os.path.join(output_dir, os.path.splitext(file)[0]+".jpg"), cropped_image)


class FishTestDataset(Dataset):
    """
    Custom PyTorch Dataset for loading test fish scale images.
    Args:
        image_dir: Directory containing images.
        transform: Transformations to apply to images.
    """
    def __init__(self, image_dir, transform=None):

        # Get the directory dataset images
        self.image_dir = image_dir

        # Get the transform methods
        self.transforms = transform


        # Image Name
        self.image_name = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]


    def __len__(self):
        # Return number of images in dataset
        return len(self.image_name)

    def __getitem__(self, index):
        # Load image and apply transforms
        img_path = os.path.join(self.image_dir, str(self.image_name[index]))
        image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        return image, self.image_name[index]
        

parser = argparse.ArgumentParser()
 # Parse command line arguments for input/output/model paths
parser.add_argument("raw_dir", help="directory of raw scale images to inference on")
parser.add_argument("out_dir", help="where to save the corresponding predictions")
parser.add_argument("model_path", help="where to find trained model weights")
args = parser.parse_args()

# Preprocess raw images: crop and pad, save to output directory
data_dir = os.path.join(args.out_dir, "cropped")
preprocess_folder(args.raw_dir, data_dir)

# Define image transformations for inference
data_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
# Create test dataset and dataloader
test_dataset = FishTestDataset( data_dir, data_transforms)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False, drop_last=False)

# Set device for inference (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
model = resnet18(num_classes = 5)

# Load model - TODO
model.load_state_dict(torch.load(args.model_path))

model.eval()    
model.to(device)

import torch

# Prepare output directory and results file
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
output_path = os.path.join(args.out_dir, "inference_results.csv")
file = open(output_path, 'w')
file.write("Image Name, Predicted Age\n")

# Run inference on test images and write predictions to CSV
for images, img_path in tqdm(test_loader):
    images = images.to(device)
    outputs = model(images)
    outputs = torch.squeeze(outputs)
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().detach().numpy()
    for i in range(preds.shape[0]):
        age = str(preds[i])
        if(preds[i] ==4):
            age = "4+"
        file.write("%s,%s\n"%(img_path[i],age))
file.close()
