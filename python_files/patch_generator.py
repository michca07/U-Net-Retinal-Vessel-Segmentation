import numpy as np

class PatchGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_non_overlapping_patches(image_set, mask_set, patch_size):
        """
        Generate non-overlapping patches from the images and their masks.

        Args:
            image_set (numpy.ndarray): Input image dataset of shape (num_images, height, width, channels).
            mask_set (numpy.ndarray): Corresponding mask dataset of shape (num_images, height, width).
            patch_size (int): Size of the patches (assumes square patches).

        Returns:
            numpy.ndarray: Dataset of patches extracted from the images.
            numpy.ndarray: Dataset of patches extracted from the masks.
        """
        if len(image_set.shape) != 4 or image_set.shape[-1] != 3:
            raise ValueError("Input image dataset must have shape (num_images, height, width, channels).")
        if len(mask_set.shape) != 3:
            raise ValueError("Input mask dataset must have shape (num_images, height, width).")
        if patch_size <= 0 or patch_size > image_set.shape[1]:
            raise ValueError("Invalid patch size. Patch size must be greater than 0 and less than or equal to image size.")

        # Calculate the number of patches that can fit in each dimension
        num_patches_height = image_set.shape[1] // patch_size
        num_patches_width = image_set.shape[2] // patch_size

        # Initialize lists to store patches
        patches_images = []
        patches_masks = []

        # Extract non-overlapping patches from images and masks
        for i in range(image_set.shape[0]):
            for h in range(num_patches_height):
                for w in range(num_patches_width):
                    # Define the coordinates for the patch
                    start_h = h * patch_size
                    end_h = (h + 1) * patch_size
                    start_w = w * patch_size
                    end_w = (w + 1) * patch_size

                    # Extract patches from the image and mask
                    patch_image = image_set[i, start_h:end_h, start_w:end_w, :]
                    patch_mask = mask_set[i, start_h:end_h, start_w:end_w]

                    # Append the patches to the lists
                    patches_images.append(patch_image)
                    patches_masks.append(patch_mask)

        # Convert lists to numpy arrays
        patches_images = np.array(patches_images)
        patches_masks = np.array(patches_masks)

        return patches_images, patches_masks

    @staticmethod
    def generate_overlapping_patches(image_set, mask_set, patch_size, stride):
        """
        Generate overlapping patches from the images and their masks.

        Args:
            image_set (numpy.ndarray): Input image dataset of shape (num_images, height, width, channels).
            mask_set (numpy.ndarray): Corresponding mask dataset of shape (num_images, height, width, channels).
            patch_size (int): Size of the patches (assumes square patches).
            stride (int): Stride for patch extraction.

        Returns:
            numpy.ndarray: Dataset of patches extracted from the images.
            numpy.ndarray: Dataset of patches extracted from the masks.
        """
        if len(image_set.shape) != 4 or image_set.shape[-1] != 3:
            raise ValueError("Input image dataset must have shape (num_images, height, width, channels).")
        if len(mask_set.shape) != 4 or mask_set.shape[-1] != 1:
            raise ValueError("Input mask dataset must have shape (num_images, height, width, 1).")
        if patch_size <= 0 or patch_size > image_set.shape[1] or stride <= 0:
            raise ValueError("Invalid patch size or stride. Patch size and stride must be greater than 0.")
        if patch_size > stride:
            raise ValueError("Patch size must be less than or equal to the stride for overlapping patches.")

        # Initialize lists to store patches
        patches_images = []
        patches_masks = []

        # Extract overlapping patches from images and masks
        for i in range(image_set.shape[0]):
            for h in range(0, image_set.shape[1] - patch_size + 1, stride):
                for w in range(0, image_set.shape[2] - patch_size + 1, stride):
                    # Define the coordinates for the patch
                    start_h = h
                    end_h = h + patch_size
                    start_w = w
                    end_w = w + patch_size

                    # Extract patches from the image and mask
                    patch_image = image_set[i, start_h:end_h, start_w:end_w, :]
                    patch_mask = mask_set[i, start_h:end_h, start_w:end_w, :]

                    # Append the patches to the lists
                    patches_images.append(patch_image)
                    patches_masks.append(patch_mask)

        # Convert lists to numpy arrays
        patches_images = np.array(patches_images)
        patches_masks = np.array(patches_masks)

        return patches_images, patches_masks
