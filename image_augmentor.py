import cv2
import numpy as np

class CustomAugmentation:
    @staticmethod
    def blend_images_with_masks(gray_images, masks, alpha):
        """
        Blend gray images with binary masks using a specified alpha value.

        Args:
            gray_images (numpy.ndarray): Input gray image dataset with shape (N, H, W, 1).
            masks (numpy.ndarray): Corresponding binary mask dataset with shape (N, H, W).
            alpha (float): Blending parameter between [0, 1].

        Returns:
            numpy.ndarray: Blended image dataset with shape (N, H, W, 1).
        """
        blended_images = (gray_images + alpha * masks[..., None]) / (1 + alpha)
        return blended_images

    @staticmethod
    def blend_gray_images_with_masks(X_train, Y_train, alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0]):
        """
        Blend gray images with binary masks using different alpha values.

        Args:
            X_train (numpy.ndarray): Input gray image dataset with shape (N, H, W, 1).
            Y_train (numpy.ndarray): Corresponding binary mask dataset with shape (N, H, W).

        Returns:
            numpy.ndarray: Blended image dataset with shape (5N, H, W, 1).
            numpy.ndarray: Blended mask dataset with shape (5N, H, W).
        """
        # Initialize numpy arrays to store blended images and masks
        blended_images_dataset = np.zeros((len(alpha_values) * len(X_train), X_train.shape[1], X_train.shape[2], 1))
        blended_masks_dataset = np.zeros((len(alpha_values) * len(X_train), Y_train.shape[1], Y_train.shape[2]))

        # Apply blending for each alpha value
        for i, alpha in enumerate(alpha_values):
            # Blend images with masks
            blended_images = CustomAugmentation.blend_images_with_masks(X_train, Y_train, alpha)

            # Fill the blended images and masks arrays
            blended_images_dataset[i * len(X_train):(i + 1) * len(X_train), :, :, :] = blended_images
            blended_masks_dataset[i * len(X_train):(i + 1) * len(X_train), :, :] = Y_train

        return blended_images_dataset, blended_masks_dataset

    @staticmethod
    def custom_augmentation(image, mask):
        """
        Apply custom data augmentation to a gray image and its corresponding mask.

        Args:
            image (numpy.ndarray): Input gray image.
            mask (numpy.ndarray): Corresponding mask.

        Returns:
            numpy.ndarray: Augmented gray image.
            numpy.ndarray: Augmented mask.
        """
        if len(image.shape) != 2:
            raise ValueError("Input image must be a grayscale image with shape (height, width).")
        if image.shape != mask.shape:
            raise ValueError("Input image and mask must have the same dimensions.")

        # Randomly select an augmentation type
        augmentation_type = np.random.choice(['rotate', 'flip_vertical', 'flip_horizontal', 'zoom_in', 'shear', 'blur'])

        if augmentation_type == 'rotate':
            # Randomly rotate the image and mask
            angle = np.random.randint(1, 360)
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            augmented_image = cv2.warpAffine(image, M, (cols, rows))
            augmented_mask = cv2.warpAffine(mask, M, (cols, rows))

        elif augmentation_type == 'flip_vertical':
            # Flip the image and mask vertically
            augmented_image = cv2.flip(image, 0)
            augmented_mask = cv2.flip(mask, 0)

        elif augmentation_type == 'flip_horizontal':
            # Flip the image and mask horizontally
            augmented_image = cv2.flip(image, 1)
            augmented_mask = cv2.flip(mask, 1)

        elif augmentation_type == 'zoom_in':
            # Randomly zoom in the image and mask
            scale = np.random.uniform(1.1, 1.5)
            augmented_image = cv2.resize(image, None, fx=scale, fy=scale)
            augmented_mask = cv2.resize(mask, None, fx=scale, fy=scale)

            rows, cols = augmented_image.shape
            crop_x = int((cols - image.shape[1]) / 2)
            crop_y = int((rows - image.shape[0]) / 2)
            augmented_image = augmented_image[crop_y:crop_y + image.shape[0], crop_x:crop_x + image.shape[1]]
            augmented_mask = augmented_mask[crop_y:crop_y + image.shape[0], crop_x:crop_x + image.shape[1]]

        elif augmentation_type == 'shear':
            # Apply shear transformation to the image and mask
            shear_range = np.random.uniform(0.1, 0.3)
            shear_matrix = np.array([[1, shear_range, 0], [0, 1, 0]], dtype=np.float32)
            augmented_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))
            augmented_mask = cv2.warpAffine(mask, shear_matrix, (mask.shape[1], mask.shape[0]))
            
        elif augmentation_type == 'blur':
            # Apply Gaussian blur to the image
            blur_amount = np.random.uniform(1, 5)
            augmented_image = cv2.GaussianBlur(image, (int(blur_amount)*2 + 1, int(blur_amount)*2 + 1), 0)
            augmented_mask = mask

        return augmented_image, augmented_mask

    def apply_custom_augmentation(self, image_set, mask_set, augmentation_factor=2):
        """
        Apply custom data augmentation to a dataset of gray images and their corresponding masks.

        Args:
            image_set (numpy.ndarray): Input gray image dataset.
            mask_set (numpy.ndarray): Corresponding mask dataset.
            augmentation_factor (int): Factor by which to increase the dataset size (default is 2).

        Returns:
            numpy.ndarray: Enlarged gray image dataset.
            numpy.ndarray: Enlarged mask dataset.
        """
        if len(image_set.shape) != 4 or image_set.shape[-1] != 1:
            raise ValueError("Input dataset must be a grayscale image dataset with shape (num_images, height, width, 1).")
        if image_set.shape[0] != mask_set.shape[0]:
            raise ValueError("Number of images and masks in the dataset must be the same.")

        augmented_images = []
        augmented_masks = []

        for i in range(image_set.shape[0]):
            # Remove the singleton dimension
            img = image_set[i].squeeze()  
            msk = mask_set[i]

            if img.shape != msk.shape:
                raise ValueError(f"Image and mask dimensions do not match for sample {i + 1} in the dataset.")

            aug_images = []
            aug_masks = []
            for _ in range(augmentation_factor):
                aug_img, aug_msk = self.custom_augmentation(img, msk)
                aug_images.append(aug_img)
                aug_masks.append(aug_msk)

            augmented_images.extend(aug_images)
            augmented_masks.extend(aug_masks)

        augmented_images = np.array(augmented_images)[:, :, :, np.newaxis]  # Add singleton dimension back
        augmented_masks = np.array(augmented_masks)

        enlarged_image_set = np.concatenate((image_set, augmented_images), axis=0)
        enlarged_mask_set = np.concatenate((mask_set, augmented_masks), axis=0)

        return enlarged_image_set, enlarged_mask_set
