# Pneumonia_Detection using Deep Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Exploration](#data-exploration)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architectures](#model-architectures)
6. [Results](#results)
7. [Model Comparison](#model-comparison)
8. [Conclusion](#conclusion)

## 📋 Project Overview

This project implements a deep learning solution for pneumonia detection from chest X-ray images. The goal is to classify chest X-ray images into two categories:
- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Pneumonia-affected lungs

The project explores multiple CNN architectures and compares their performance, ultimately selecting the best-performing model for pneumonia detection.

## 📊 Dataset

The dataset used is the [Chest X-ray Images](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) from Kaggle, downloaded using the kagglehub library.

### Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Dataset Statistics
- **Training Set**:
  - NORMAL: 1,341 images
  - PNEUMONIA: 3,875 images
  - Total: 5,216 images

- **Test Set**:
  - NORMAL: 234 images
  - PNEUMONIA: 390 images
  - Total: 624 images

### Class Distribution
The dataset shows class imbalance with pneumonia cases being more prevalent than normal cases (~3:1 ratio in training set).

## Data Exploration

The project includes comprehensive data exploration:
- Visual inspection of random sample images from both classes
- Distribution analysis of training and test sets
- Sample visualization with proper labeling

## Data Preprocessing

### Image Transformations

#### Basic Transform (for validation/test)
```python
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

#### Augmented Transform (for training)
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomAutocontrast(p=0.3),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

## 📐 Model Architectures

### 1. CNN2 - Basic 2-Layer CNN
**Architecture:**
- 2 Convolutional layers (32, 64 filters)
- ReLU activation
- MaxPooling layers
- Fully connected classifier

### 2. CNN2_BN_D - 2-Layer CNN with Regularization
**Architecture:**
- 2 Convolutional layers with Batch Normalization
- Dropout (0.2) for regularization
- Improved generalization over basic CNN

**Improvements:**
- Batch normalization for stable training
- Dropout to prevent overfitting

### 3. CNN3_BN_D - 3-Layer CNN with Regularization
**Architecture:**
- 3 Convolutional layers (32, 64, 128 filters)
- Batch normalization after each conv layer
- Dropout (0.3) in classifier
- Deeper feature extraction

**Improvements:**
- More complex feature learning
- Better representation capability

### 4. Pretrained ResNet18 (Final Model)
**Architecture:**
- ResNet18 backbone pretrained on ImageNet
- Modified first layer for grayscale input (1 channel)
- Fine-tuned for binary classification
- Transfer learning approach

## Training Details

### Training Configuration
- **Batch Size:** 32
- **Optimizer:** Adam
- **Learning Rate:** 1e-3 (custom CNNs), 1e-4 (pretrained)
- **Loss Function:** CrossEntropyLoss
- **Early Stopping:** Patience of 5 epochs
- **Device:** CUDA if available, else CPU

### Training Features
- Early stopping to prevent overfitting
- Model checkpointing (saves best model based on validation loss)
- Learning rate scheduling for pretrained model
- Comprehensive logging of training/validation metrics

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| CNN2 | 0.8654 | 0.8621 | 0.8654 | 0.8631 |
| CNN2_BN_D | 0.9038 | 0.9015 | 0.9038 | 0.9021 |
| CNN3_BN_D | 0.9231 | 0.9211 | 0.9231 | 0.9218 |
| **Pretrained ResNet18** | **0.9712** | **0.9710** | **0.9712** | **0.9710** |

### Key Findings

1. **Regularization Impact:** Batch normalization and dropout significantly improved performance
2. **Transfer Learning Success:** Pretrained ResNet18 achieved the best results with 97.12% accuracy
3. **Generalization:** The final model shows excellent performance across all metrics

### Confusion Matrix Analysis
The confusion matrices reveal:
- High true positive rate for pneumonia detection
- Low false negative rate (critical for medical applications)
- Excellent overall classification performance

## Model Comparison

### Why Pretrained ResNet18 Performed Best?

### Performance Metrics Explanation:

- **Accuracy (97.12%):** Overall correctness of predictions
- **Precision (97.10%):** Proportion of positive predictions that were correct
- **Recall (97.12%):** Proportion of actual positives correctly identified
- **F1 Score (97.10%):** Harmonic mean of precision and recall

## 📌 Conclusion

The pneumonia detection system successfully achieves high accuracy (97.12%) using a pretrained ResNet18 model. Key success factors include:

1. **Effective Data Augmentation:** Improved model robustness
2. **Transfer Learning:** Leveraged pretrained features for medical imaging
3. **Proper Regularization:** Prevented overfitting while maintaining performance
4. **Comprehensive Evaluation:** Multiple metrics ensure reliable assessment

### Clinical Relevance
- High sensitivity (recall) is crucial for pneumonia detection to minimize missed cases
- The model's 97.12% accuracy makes it a valuable diagnostic aid
- Low false negative rate reduces risk of undiagnosed pneumonia cases

### Future Improvements
1. **Dataset Expansion:** Include more diverse X-ray images
2. **Multi-class Classification:** Distinguish between bacterial and viral pneumonia
3. **Clinical Validation:** Test on real-world clinical data
4. **Deployment:** Develop user-friendly interface for medical professionals

