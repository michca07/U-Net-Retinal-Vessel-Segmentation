
# Retinal Vessel Segmentation

Retinal vessel segmentation is a critical task in medical image analysis, playing a pivotal role in the diagnosis and management of various eye conditions, including diabetic retinopathy and hypertensive retinopathy. Accurate segmentation of retinal vessels provides valuable insights into the vascular architecture of the retina, enabling early detection and monitoring of ocular diseases.

### Importance

The importance of retinal vessel segmentation lies in its ability to assist clinicians in identifying abnormalities, assessing disease progression, and making informed decisions for patient care. Early detection of anomalies in retinal vessels can lead to timely intervention and improved patient outcomes.

### Available Methods

Several methods have been proposed for retinal vessel segmentation, ranging from traditional image processing techniques to advanced deep learning approaches. Classical methods often involve the application of filters, thresholding, and morphological operations. On the other hand, recent advancements in deep learning, particularly convolutional neural networks (CNNs), have shown promising results in capturing complex patterns and improving segmentation accuracy.

### U-Net

In this project, I leverage the power of U-Net, a convolutional neural network architecture specifically designed for semantic segmentation tasks. U-Net has demonstrated success in medical image analysis, including retinal vessel segmentation. Its architecture, featuring a contracting path, a bottleneck, and an expansive path, allows for effective feature extraction and precise localization of retinal vessels.

By adopting U-Net, I aim to build a robust and accurate model that surpasses traditional methods and contributes to the ongoing advancements in retinal image analysis.

## Objectives

1. **Segmentation of RGB Retinal Images:**
   Develop a U-Net model to accurately segment retinal vessels in RGB images. The model will distinguish between vessel and non-vessel regions, enhancing our understanding of retinal vascular structure.

2. **Optimization Based on Dice Coefficient:**
   Optimize segmentation performance using the Dice coefficient as a key metric. Achieving a high Dice coefficient indicates precise vessel segmentation, contributing to the model's overall accuracy.

3. **U-Net Architecture Implementation:**
   Implement a U-Net architecture for retinal vessel segmentation. The model, built using TensorFlow and Keras, features an encoder-decoder structure with skip connections. This design facilitates effective feature extraction and localization of retinal vessels.

## Preprocessing

### Image Preprocessing

In preparation for training the U-Net model, various preprocessing techniques are applied to enhance the quality and features of retinal images.

1. **RGB to Gray Conversion:**
   RGB images are converted to grayscale using a weighted sum method to preserve important information. The conversion is performed with the following ratios: R=0, G=0.7, B=0.3. This ensures that the green channel contributes the most to the grayscale representation, aligning with the significance of vessels in the green channel.

2. **Grid-based CLAHE (Contrast Limited Adaptive Histogram Equalization):**
   Contrast enhancement is achieved through a grid-based CLAHE approach. The image is divided into (8,8) grids, and CLAHE is applied independently to each grid. This grid-wise application of CLAHE helps prevent over-amplification of noise in homogeneous regions, improving the overall visibility of vessels.

3. **Gamma Correction:**
   Gamma correction is employed to adjust the overall brightness of the images. This nonlinear operation involves raising the intensity values to a certain power (gamma). It enhances contrast and fine-tunes the overall appearance of the retinal images, contributing to better feature extraction during the model training process.

The combination of these preprocessing steps aims to provide the U-Net model with well-prepared input data, promoting effective learning and improving the model's ability to accurately segment retinal vessels.

## Evaluation

The performance of the trained U-Net model is rigorously evaluated using a set of key metrics, providing a comprehensive understanding of its segmentation accuracy and effectiveness.

- **Average F1 Score:**
  The F1 score is a measure that combines precision and recall, providing a balanced assessment of the model's ability to correctly identify vessel pixels while minimizing false positives and false negatives. The average F1 score is computed across the entire dataset.

- **Average AUC (Area Under the Curve):**
  The AUC is a metric commonly used in binary classification problems. It quantifies the model's ability to distinguish between vessel and non-vessel regions. The average AUC provides an aggregated assessment of the model's performance across different subsets of the data.

- **Average Accuracy:**
  Accuracy measures the overall correctness of the model's predictions. The average accuracy is computed by dividing the total number of correctly classified pixels by the total number of pixels in the dataset, offering a holistic view of the model's segmentation performance.

- **Average Dice Coefficient:**
  The Dice coefficient is another measure of segmentation accuracy, representing the overlap between the predicted and ground truth masks. The average Dice coefficient provides a robust evaluation of the U-Net model's ability to precisely delineate retinal vessels.

These metrics collectively offer a thorough analysis of the U-Net model's performance, enabling insights into its strengths and areas for potential improvement. The goal is to achieve high scores across these metrics, indicating accurate and reliable retinal vessel segmentation.

## Results

The performance results of the U-Net model on both the training and test sets are presented below:

| Metric            | Training Set    | Test Set    |
|-------------------|-----------------|-------------|
| Average F1 Score  | 0.87028         | 0.81064     |
| Average AUC       | 0.91988         | 0.88689     |
| Average Accuracy  | 0.97842         | 0.97003     |
| Average Dice Coeff| 0.87029         | 0.81066     |


![Segmentation Results](images/Example.png)

The table summarizes the average F1 score, average AUC, average accuracy, and average Dice coefficient for both the training and test sets. The provided image, 'Example.png,' showcases a visual representation of the segmentation results, providing additional insights into the model's performance.
