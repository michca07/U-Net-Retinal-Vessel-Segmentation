import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _check_input_shape(image_set):
        if len(image_set.shape) != 4 or image_set.shape[-1] != 1:
            raise ValueError("Input dataset must be a grayscale image dataset with shape (num_images, height, width, 1).")

    @staticmethod
    def _check_uint8_dtype(image_set):
        if image_set.dtype != np.uint8:
            raise ValueError("Input dataset must have dtype np.uint8.")

    def rgb_to_gray_weighted(self, images):
        self._check_input_shape(images)
        gray_images = np.dot(images[...,:3], [0.0, 0.7, 0.3]).reshape(images.shape[:-1] + (1,))
        return gray_images

    def apply_histogram_equalization(self, image_set):
        self._check_input_shape(image_set)
        self._check_uint8_dtype(image_set)
        
        equalized_images = np.empty(image_set.shape)
        for i in range(image_set.shape[0]):
            equalized_image = np.empty_like(image_set[i])
            for c in range(image_set.shape[3]):
                equalized_image[..., c] = cv2.equalizeHist(image_set[i, ..., c])
            equalized_images[i] = equalized_image
        return equalized_images

    def apply_contrast_enhancement(self, image_set):
        self._check_input_shape(image_set)
        self._check_uint8_dtype(image_set)
        
        enhanced_images = np.empty(image_set.shape)
        for i in range(image_set.shape[0]):
            enhanced_image = np.empty_like(image_set[i])
            for c in range(image_set.shape[3]):
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced_image[..., c] = clahe.apply(image_set[i, ..., c])
            enhanced_images[i] = enhanced_image
        return enhanced_images

    def grid_based_enhance_contrast(self, image_set, grid_size=(4, 4), method='clahe', clip_limit=2.0):
        self._check_input_shape(image_set)
        self._check_uint8_dtype(image_set)
        
        num_images, rows, cols, _ = image_set.shape
        sub_image_height = rows // grid_size[0]
        sub_image_width = cols // grid_size[1]
        enhanced_images = np.zeros_like(image_set, dtype=np.float32)

        for k in range(num_images):
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    start_row = i * sub_image_height
                    end_row = (i + 1) * sub_image_height
                    start_col = j * sub_image_width
                    end_col = (j + 1) * sub_image_width
                    sub_image = image_set[k, start_row:end_row, start_col:end_col, :]

                    if method == 'clahe':
                        clahe = cv2.createCLAHE(clipLimit=clip_limit)
                        enhanced_sub_image = clahe.apply(sub_image.squeeze())  # Remove singleton dimension
                    elif method == 'hist':
                        enhanced_sub_image = cv2.equalizeHist(sub_image.squeeze())
                    else:
                        raise ValueError("Invalid method. Use 'clahe' or 'hist'.")

                    enhanced_sub_image = np.expand_dims(enhanced_sub_image, axis=-1)
                    enhanced_images[k, start_row:end_row, start_col:end_col, :] = enhanced_sub_image

        return enhanced_images

    def apply_gamma_correction(self, image_set, gamma=1.0):
        self._check_input_shape(image_set)
        self._check_uint8_dtype(image_set)
        
        corrected_images = np.empty(image_set.shape)
        for i in range(image_set.shape[0]):
            corrected_image = np.power(image_set[i], gamma)
            corrected_image = (corrected_image / np.max(corrected_image)) * 255
            corrected_images[i] = corrected_image
        return corrected_images

    def apply_unsharp_masking(self, image_set, alpha=1.5):
        self._check_input_shape(image_set)
        self._check_uint8_dtype(image_set)
        
        sharpened_images = np.empty(image_set.shape)
        laplacian_kernel = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])

        for i in range(image_set.shape[0]):
            edges = cv2.filter2D(image_set[i], -1, laplacian_kernel)
            mask = alpha * edges
            blurred_image = cv2.GaussianBlur(image_set[i], (5, 5), 1.5)
            sharpened_image = cv2.add(blurred_image, mask)
            sharpened_images[i] = sharpened_image

        return sharpened_images
