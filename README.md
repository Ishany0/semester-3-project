# AI-Powered Medical Diagnosis using Deep Learning and Explainable AI

## Overview

This project presents a generalized AI-based diagnostic framework for disease detection using medical imaging data. The system leverages Transfer Learning with ResNet50 to classify medical images and generate interpretable predictions through Explainable AI techniques such as Grad-CAM and LIME.

The framework was developed to improve diagnostic accuracy, reduce manual screening effort, and provide transparent predictions that healthcare professionals can interpret and trust.

## Features

* Medical image classification using Deep Learning
* Transfer Learning with ResNet50 pretrained on ImageNet
* COVID-19 detection from X-Ray and CT images
* Pneumonia detection from Chest X-Rays
* Blood Cell Classification across 8 cell categories
* Explainable AI using:

  * Grad-CAM Heatmaps
  * LIME Explanations
* Fine-tuning strategy for improved performance
* ROC-AUC, F1-Score, Sensitivity and Specificity evaluation
* Optimized TensorFlow data pipeline

---

## Datasets

### 1. Chest X-Ray Pneumonia Dataset

* Task: Pneumonia vs Normal Classification
* Images: 5,863
* Format: JPEG
* Source: Kaggle

### 2. COVID-19 CT and X-Ray Dataset

* Task: COVID vs Non-COVID Classification
* Images: 17,099
* Formats: JPG, JPEG, PNG
* Source: Mendeley Data

### 3. Blood Cell Images Dataset

* Task: Multi-class Blood Cell Classification

* Images: 12,500

* Classes:

  * Basophil
  * Eosinophil
  * Erythroblast
  * Immature Granulocyte (IG)
  * Lymphocyte
  * Monocyte
  * Neutrophil
  * Platelet

* Source: Kaggle

---

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn
* Grad-CAM
* LIME

---

## Model Architecture

### Base Network

* ResNet50 (Pretrained on ImageNet)

### Classification Head

* Global Average Pooling
* Dense Layer (256 units)
* Dense Layer (128 units)
* Dense Layer (64 units)
* Dropout (0.3)
* L2 Regularization
* Softmax Output Layer

---

## Data Preprocessing

The following preprocessing steps were applied:

1. Image resizing to 224 × 224
2. RGB color conversion
3. Normalization using ImageNet preprocessing
4. Data augmentation

   * Random flips
   * Rotations
   * Zoom operations
5. Dataset caching and prefetching
6. Learning rate scheduling using ReduceLROnPlateau

---

## Training Strategy

### Phase 1

* Freeze entire ResNet50 backbone
* Train custom classifier head

### Phase 2

* Unfreeze last 30 layers
* Fine-tune using a low learning rate

### Optimizer

* Adam Optimizer

### Loss Function

* Sparse Categorical Crossentropy

---

## Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Sensitivity
* Specificity
* Confusion Matrix

---

## Explainable AI

### Grad-CAM

Gradient-weighted Class Activation Mapping was used to visualize regions of medical images influencing model predictions.

Benefits:

* Improves transparency
* Highlights infected regions
* Helps clinicians verify model reasoning

### LIME

Local Interpretable Model-Agnostic Explanations were used to explain predictions for individual samples.

Benefits:

* Instance-level interpretability
* Better understanding of model behavior
* Increased trust in predictions

---

## Project Workflow

```text
Medical Datasets
       │
       ▼
Data Preprocessing
       │
       ▼
Train / Validation Split
       │
       ▼
Transfer Learning (ResNet50)
       │
       ▼
Fine-Tuning
       │
       ▼
Model Evaluation
       │
       ▼
Grad-CAM & LIME
       │
       ▼
Deployable Diagnostic Model
```

## Results

* High classification accuracy across medical imaging datasets.
* Strong ROC-AUC performance for disease detection.
* Improved F1-score through threshold optimization.
* Grad-CAM visualizations successfully highlighted disease-related regions.
* Demonstrated potential for scalable AI-assisted clinical screening.

## Ethical Considerations

Potential limitations include:

* Dataset bias and class imbalance
* False positives leading to unnecessary follow-up
* False negatives potentially missing disease cases
* Limited demographic diversity in training datasets

These challenges should be addressed before real-world clinical deployment.

## Future Work

* Expand datasets across hospitals and demographics
* Multi-class disease classification
* Integration with web-based diagnostic systems
* Federated learning for privacy-preserving training
* Real-time deployment in healthcare environments

## Author

**Ishanya Sharma**

B.E. AIML

## Disclaimer

This project is intended for educational and research purposes only and should not be used as a substitute for professional medical diagnosis.
