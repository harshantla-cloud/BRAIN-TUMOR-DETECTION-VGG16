# 🧠 Brain Tumor Detection using VGG16

A deep learning-based Brain Tumor Detection and Classification System using the VGG16 convolutional neural network architecture to classify MRI brain images into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

## 📌 Project Overview

This project applies Deep Learning, Transfer Learning, and Computer Vision techniques to MRI brain image classification. The VGG16 architecture is used to learn visual features from MRI images and classify them into the four target classes.

The project was developed using Python, TensorFlow, Keras, and Google Colab.

## 🎯 Objectives

- Detect and classify brain tumors from MRI images.
- Apply VGG16 transfer learning to medical image classification.
- Preprocess MRI images for deep learning.
- Evaluate the trained model using standard classification metrics.
- Predict the class of new MRI images with a confidence score.

## 🏗️ Model Workflow

```text
MRI Image
    ↓
Image Preprocessing
    ↓
Resize to 128 × 128
    ↓
Pixel Normalization
    ↓
VGG16 Feature Extraction
    ↓
Classification Layers
    ↓
Prediction
    ↓
Glioma / Meningioma / Pituitary / No Tumor
```

## 🧠 Model Architecture

The project uses VGG16 with a transfer learning approach. The pretrained convolutional architecture is used to extract useful image features, followed by classification layers for the four MRI categories.

## 📂 Dataset Classes

```text
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

> The MRI dataset is not included in this repository unless explicitly added separately.

## 🔧 Technologies Used

- Python
- TensorFlow
- Keras
- VGG16
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Google Drive

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/harshantla-cloud/BRAIN-TUMOR-DETECTION-VGG16.git
cd BRAIN-TUMOR-DETECTION-VGG16
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Load the Trained Model

The trained model is saved in Keras format:

```python
from tensorflow.keras.models import load_model

model = load_model(
    '/content/drive/MyDrive/MRI Images/brain_tumor_model.keras'
)

print("Model loaded successfully!")
```

## 🔍 Prediction

Example prediction workflow:

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

class_labels = [
    'pituitary',
    'glioma',
    'notumor',
    'meningioma'
]

img = load_img(
    image_path,
    target_size=(128, 128)
)

img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)

predicted_class_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_class_index]

print("Prediction:", class_labels[predicted_class_index])
print(f"Confidence: {confidence * 100:.2f}%")
```

## 📊 Model Evaluation

The model can be evaluated using:

- Accuracy
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC

### Results

Update the following table with the actual values obtained from your trained model:

| Metric | Result |
|---|---:|
| Accuracy | Add actual result |
| Precision | Add actual result |
| Recall | Add actual result |
| F1-Score | Add actual result |
| AUC | Add actual result |

## 📁 Project Structure

```text
BRAIN-TUMOR-DETECTION-VGG16/
│
├── brain_tumor_detection_using_deep_learning.ipynb
├── brain_tumor_model.keras
├── requirements.txt
└── README.md
```

## 💡 Key Features

- VGG16-based deep learning model
- Transfer learning
- Four-class MRI image classification
- Image resizing and normalization
- Model evaluation
- Confidence-based prediction
- Keras model saving and loading
- Google Colab-compatible workflow

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis or clinical decision-making. MRI results should be evaluated by qualified healthcare professionals.

## 👨‍💻 Author

**Harsh**

B.Tech Computer Science Engineering  
Global Institute of Technology and Management

GitHub: https://github.com/harshantla-cloud

LinkedIn: https://www.linkedin.com/in/harsh-5694b13ab

## ⭐ Acknowledgement

This project demonstrates the application of Deep Learning, Transfer Learning, Computer Vision, and VGG16 to brain MRI image classification.

If you find this project useful, consider giving the repository a ⭐.
