# Plant Disease Detection using CNN

## Overview
This project aims to detect plant diseases from images using a Convolutional Neural Network (CNN) model. The CNN is trained on a labeled dataset of healthy and diseased plant leaves. The goal is to classify each image into one of the predefined categories, such as different types of diseases or healthy leaves.

## Dataset
The dataset contains images of plant leaves categorized into multiple classes, such as:
- **Healthy** leaves
- **Diseased** leaves (with specific diseases like Apple Scab, Black Rot, Leaf Curl, etc.)

Images are preprocessed to a standard size of 150x150 pixels, normalized, and augmented to improve model robustness. The dataset is split into training and testing sets.

## Model Architecture
The model is a sequential Convolutional Neural Network (CNN) consisting of:
1. **Convolutional Layers**: Extract features from the images.
   - 1st Conv Layer: 32 filters of size (3x3), ReLU activation
   - 2nd Conv Layer: 64 filters of size (3x3), ReLU activation
   - 3rd Conv Layer: 128 filters of size (3x3), ReLU activation
2. **MaxPooling Layers**: Reduce the spatial dimensions of the feature maps.
3. **Flatten Layer**: Convert the 2D matrices into a 1D vector.
4. **Dense Layers**: Fully connected layers with ReLU activation.
   - Includes Dropout to prevent overfitting.
5. **Output Layer**: Softmax activation for multi-class classification.

The model is compiled using the Adam optimizer and categorical cross-entropy loss.

## Training
The model is trained with data augmentation techniques such as:
- Random rotations, zooming, shearing, and horizontal flipping.

The model uses early stopping and checkpoints to save the best-performing model during training.

## Dependencies
To install dependencies, run:
```bash
pip install -r requirements.txt
```

## Running the Model
To train and evaluate the model:
```bash
python train.py
```

## Model Inference
You can use the trained model to predict the disease for new images. Use the `predict_image` function to load an image and predict the disease.

Example:
```python
predict_image('path_to_image', model, class_indices)
```

## Accuracy
The model achieves an accuracy of approximately **92%** on the test dataset.

## How to Use
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Test the model:
   ```bash
   python evaluate.py
   ```
5. Predict disease for a new image:
   ```bash
   python predict.py --image path_to_image
   ```

---

Feel free to modify this template to better fit your project!
