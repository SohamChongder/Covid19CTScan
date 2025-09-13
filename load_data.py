import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

def load_data(img_rows, img_cols):
    images = []
    labels = []

    # Path to the folder containing the images
    data_folder1 = "/Users/sohamchongder/Desktop/Medical Imaging project/archive/COVID"
    # Path to the folder containing the images
    data_folder2 = "/Users/sohamchongder/Desktop/Medical Imaging project/archive/non-COVID"
    # Iterate through each image file in the folder
    for filename in os.listdir(data_folder1):
        # print(filename)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Load the image and convert it to numpy array
            img = Image.open(os.path.join(data_folder1, filename))
            img = img.convert("RGB")
            nimg=img.resize((img_rows, img_cols))
            img_array = np.array(nimg)
            # Switch RGB to BGR order 
            # img_array = img_array[:, :, :, ::-1]
            # # Subtract ImageNet mean pixel   
            # img_array[:, :, :, 0] -= 103.939
            # img_array[:, :, :, 1] -= 116.779
            # img_array[:, :, :, 2] -= 123.68

            # print(img_array.shape)
            images.append(img_array)
            labels.append(1)  # Setting label as 1 for all images

    for filename in os.listdir(data_folder2):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Load the image and convert it to numpy array
            img = Image.open(os.path.join(data_folder2, filename))
            img = img.convert("RGB")
            nimg=img.resize((img_rows, img_cols))
            img_array = np.array(nimg)
            # Switch RGB to BGR order 
            # img_array = img_array[:, :, :, ::-1]
            # # Subtract ImageNet mean pixel   
            # img_array[:, :, :, 0] -= 103.939
            # img_array[:, :, :, 1] -= 116.779
            # img_array[:, :, :, 2] -= 123.68

            # print(img_array.shape)
            images.append(img_array)
            labels.append(0)  # Setting label as 1 for all images

    
    images = np.array(images)
    labels = np.array(labels)

    # Splitting data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    # return images
    return X_train, Y_train, X_val, Y_val

# Path to the folder containing the images
# data_folder = "/Users/sohamchongder/Desktop/Medical Imaging project/archive/COVID"

# Loading data
X_train, Y_train, X_val, Y_val = load_data(224,224)
print(Y_train)
print("Shape of X_train:", X_train.shape)
print("Shape of Y_train:", Y_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of Y_val:", Y_val.shape)
