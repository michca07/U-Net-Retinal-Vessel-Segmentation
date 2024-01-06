import os
import numpy as np
from skimage.io import imread

def load_dataset(images_folder, masks_folder):
    """
    Load a dataset from specified folders containing images and masks.

    Args:
        images_folder (str): Path to the folder containing image files.
        masks_folder (str): Path to the folder containing mask files.

    Returns:
        numpy.ndarray: Array of images.
        numpy.ndarray: Array of masks.

    Raises:
        ValueError: If the number of images and masks is not the same.
        FileNotFoundError: If image or mask file is not found.
        ValueError: If image or mask has a different number of channels.
    """
    try:
        # Get the list of file names in each folder
        image_files = os.listdir(images_folder)
        mask_files = os.listdir(masks_folder)

        # Sort the files to ensure they are in the same order
        image_files.sort()
        mask_files.sort()

        # Initialize empty lists to store the images and masks
        images = []
        masks = []

        # Load images and masks
        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(images_folder, img_file)
            mask_path = os.path.join(masks_folder, mask_file)

            img = imread(img_path)
            mask = imread(mask_path)

            # Check if image and mask have the same number of channels
            if img.ndim != mask.ndim:
                raise ValueError("Image and mask must have the same number of channels.")

            images.append(img)
            masks.append(mask)

        # Convert the lists to numpy arrays
        images = np.array(images)
        masks = np.array(masks)

        # Check if the number of images and masks is the same
        if len(images) != len(masks):
            raise ValueError("Number of images and masks must be the same.")

        return images, masks

    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")


