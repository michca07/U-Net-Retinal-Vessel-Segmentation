
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tensorflow.keras.optimizers import Adam

import data_loader
import image_processor
import image_augmentor
import patch_generator
import model_builder
import eval_metrics

# To ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Load the images and their masks
# Enter the right path to the images and their masks
image_set, mask_set = data_loader(images_path, masks_path)

# Flatten the mask_data to (N, H*W) to use with train_test_split
mask_set_flat = mask_set.reshape((mask_set.shape[0], -1))

# Split the datasets into training and testing sets
image_train, image_test, mask_train_flat, mask_test_flat = train_test_split(
    image_set, mask_set_flat, test_size=0.2, random_state=42)

# Reshape the flattened mask back to its original shape
mask_train = mask_train_flat.reshape((mask_train_flat.shape[0], mask_set.shape[1], mask_set.shape[2]))
mask_test = mask_test_flat.reshape((mask_test_flat.shape[0], mask_set.shape[1], mask_set.shape[2]))

# Define X_train, Y_train
X_train = image_train
Y_train = mask_train

# Define X_test, Y_test
X_test = image_test
Y_test = mask_test

# Apply the Image Processor
X_train = image_processor.ImageProcessor.rgb_to_gray_weighted(X_train)
X_train = image_processor.ImageProcessor.grid_based_enhance_contrast(X_train, grid_size=(8, 8), method='clahe')
X_train = image_processor.ImageProcessor.apply_gamma_correction(X_train, gamma=0.9)

# Apply Image Augmentor
X_train_blended, Y_train_blended = image_augmentor.CustomAugmentation.blend_gray_images_with_masks(X_train, Y_train, alpha_values = [0.0])
X_train_enlarged, Y_train_enlarged = image_augmentor.CustomAugmentation.apply_custom_augmentation(X_train_blended, Y_train_blended, augmentation_factor = 2)

# Generate non-overlapping patches
image_patch_train, mask_patch_train = patch_generator.PatchGenerator.generate_non_overlapping_patches(X_train_enlarged, Y_train_enlarged, patch_size=256)

# Build U-Net
model = model_builder.UNetBuilder.build_unet(input_shape=(256, 256, 1), dropout_rate=0.1, l2_penalty=0.001)

# Normalize the images and masks
image_patch_train = image_patch_train.astype('float32') / 255.0
mask_patch_train = mask_patch_train /255
mask_patch_train[mask_patch_train > 0.5] = 1
mask_patch_train[mask_patch_train <= 0.5] = 0

# Compile the model
model.compile(optimizer=Adam(lr=1e-2), loss='binary_crossentropy', metrics=['accuracy', eval_metrics.EvaluationMetrics.dice_coef])
model.summary()

# Fit the model
history = model.fit(image_patch_train, mask_patch_train, epochs = 150, batch_size = 32, validation_split = 0.1)

# Save the model to a file
model.save('vessel_segmentation_unet_model.h5')

# Test the model
X_test = image_processor.ImageProcessor.rgb_to_gray_weighted(X_test)
X_test = image_processor.ImageProcessor.grid_based_enhance_contrast(X_test, grid_size=(8, 8), method='clahe')
X_test = image_processor.ImageProcessor.apply_gamma_correction(X_test, gamma=1.2)
X_test_blended, Y_test_blended = image_augmentor.CustomAugmentation.blend_gray_images_with_masks(X_test, Y_test, alpha_values = [0.0])
image_patch_test, mask_patch_test = patch_generator.PatchGenerator.generate_non_overlapping_patches(X_test_blended, Y_test_blended, patch_size=256)
image_patch_test = image_patch_test.astype('float32') / 255.0
mask_patch_test = mask_patch_test /255
mask_patch_test[mask_patch_test > 0.5] = 1
mask_patch_test[mask_patch_test <= 0.5] = 0

# Evaluate on the test image-set
def calculate_metrics(model, X_test, Y_test, threshold=0.5):
    """
    Calculate various metrics for a U-Net model on a test set.

    Args:
        model (tf.keras.Model): Trained U-Net model.
        X_test (numpy.ndarray): Test dataset.
        Y_test (numpy.ndarray): Ground truth masks.
        threshold (float): Threshold for converting probabilities to binary (default is 0.5).

    Returns:
        float: Average F1 score.
        float: Average AUC score.
        float: Average accuracy.
        float: Average Dice coefficient.
    """
    num_samples = X_test.shape[0]
    f1_scores = []
    auc_scores = []
    accuracy_scores = []
    dice_coefficients = []

    for i in range(num_samples):
        # Predictions
        predictions = model.predict(np.expand_dims(X_test[i], axis=0))[0]

        # Convert predictions to binary
        binary_predictions = (predictions > threshold).astype(np.uint8)
        Y_test_binary = (Y_test[i] > 0.5).astype(np.uint8)  # Ensure ground truth is binary

        # Flatten the arrays for metrics calculation
        Y_test_flat = Y_test_binary.flatten()
        binary_predictions_flat = binary_predictions.flatten()

        # Calculate metrics
        f1 = f1_score(Y_test_flat, binary_predictions_flat)
        auc = roc_auc_score(Y_test_flat, predictions.flatten())
        accuracy = accuracy_score(Y_test_flat, binary_predictions_flat)

        # Dice coefficient
        intersection = np.sum(Y_test_flat * binary_predictions_flat)
        union = np.sum(Y_test_flat) + np.sum(binary_predictions_flat)
        dice_coefficient = (2. * intersection + 1) / (union + 1)

        # Append to lists
        f1_scores.append(f1)
        auc_scores.append(auc)
        accuracy_scores.append(accuracy)
        dice_coefficients.append(dice_coefficient)

    # Calculate averages
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(auc_scores)
    avg_accuracy = np.mean(accuracy_scores)
    avg_dice = np.mean(dice_coefficients)

    return avg_f1, avg_auc, avg_accuracy, avg_dice

avg_f1, avg_auc, avg_accuracy, avg_dice = calculate_metrics(model, image_patch_test, mask_patch_test)
print("Average F1 Score:", avg_f1)
print("Average AUC Score:", avg_auc)
print("Average Accuracy:", avg_accuracy)
print("Average Dice Coefficient:", avg_dice)















