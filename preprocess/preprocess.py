# -*- coding: utf-8 -*-
"""
@author: satvi
"""

import pandas as pd
import numpy as np
from skimage.io import imread
import os

# Load the CSV file
def load_segmentation_data(csv_path):
    """Load the segmentation data from the CSV file."""
    return pd.read_csv(csv_path)

# Function to decode RLE into a binary mask
def rle_decode(mask_rle, shape=(768, 768)):
    """
    Decodes run-length encoded (RLE) strings into a binary mask.
    
    :param mask_rle: Run-length encoded pixels as string
    :param shape: (height, width) of the mask
    :return: Numpy array of the mask
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

# Function to load images and corresponding masks
def load_image_and_mask(image_id, df, image_folder):
    """
    Loads an image and its corresponding binary mask.
    
    :param image_id: Image ID from the dataframe
    :param df: DataFrame containing the image and segmentation information
    :param image_folder: Folder where the images are stored
    :return: Image and mask arrays
    """
    # Load the image
    image_path = os.path.join(image_folder, image_id)
    image = imread(image_path)

    # Get the encoded pixels for the image
    image_masks = df[df['ImageId'] == image_id]['EncodedPixels'].values
    mask = np.zeros((768, 768), dtype=np.uint8)

    for mask_rle in image_masks:
        if pd.isna(mask_rle):
            continue
        mask += rle_decode(mask_rle)
    
    return image, mask

# Main preprocessing function
def preprocess_data(csv_path, image_folder, output_folder):
    """
    Preprocess the dataset: load images, decode RLE masks, and save the processed data.
    
    :param csv_path: Path to the CSV file with segmentation information
    :param image_folder: Path to the folder containing images
    :param output_folder: Path to the folder where processed data will be saved
    """
    # Load the segmentation data
    df = load_segmentation_data(csv_path)
    
    # Process each image
    for image_id in df['ImageId'].unique():
        image, mask = load_image_and_mask(image_id, df, image_folder)
        
        # Save the image and mask
        image_output_path = os.path.join(output_folder, f'{image_id}')
        mask_output_path = os.path.join(output_folder, f'{image_id}_mask.npy')
        
        np.save(image_output_path, image)  # Save the image (optionally save as jpg/png)
        np.save(mask_output_path, mask)    # Save the mask as .npy file
        
        print(f'Processed {image_id} and saved to {output_folder}')

# Example usage
if __name__ == "__main__":
    csv_path = 'data/train_ship_segmentations_v2.csv'
    image_folder = 'data/images/folder'
    output_folder = 'output/'
    
    preprocess_data(csv_path, image_folder, output_folder)
