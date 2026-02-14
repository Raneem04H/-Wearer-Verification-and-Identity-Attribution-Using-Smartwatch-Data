
# Biometric Classification Model using Neural Networks

## Overview

This project implements a deep learning model to classify biometric cases based on extracted features. The model is designed to support digital forensics analysis by automatically identifying biometric patterns and predicting the correct case category.

The system uses a fully connected neural network built with TensorFlow and Keras, along with proper preprocessing, scaling, and class balancing techniques to ensure reliable and accurate predictions.

---

## Objectives

* Build a neural network model for biometric classification
* Handle imbalanced datasets using class weighting
* Apply feature scaling and preprocessing
* Evaluate model performance using accuracy, loss, and confusion matrix
* Support digital forensics analysis through automated classification

---

## Dataset

The dataset contains biometric-related features and a target column:

* **Input features:** biometric measurements and extracted attributes
* **Target variable:** `DB_Case` (classification label)
* Non-biometric attributes such as TimeStamp and demographic data were removed to improve model relevance.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Joblib

---

## Model Architecture

The neural network consists of:

* Dense layer (128 neurons, ReLU)

* Batch Normalization

* Dropout (0.25)

* Dense layer (64 neurons, ReLU)

* Batch Normalization

* Dropout (0.25)

* Dense layer (32 neurons, ReLU)

* Dropout (0.25)

* Output layer (Softmax activation)

Loss function: sparse_categorical_crossentropy
Optimizer: Adam

Regularization techniques used:

* Dropout
* Batch Normalization
* Early Stopping
* ReduceLROnPlateau
* Class Weight Balancing

---

## Training Process

Steps performed:

1. Load and clean dataset
2. Remove irrelevant columns
3. Handle missing values
4. Encode target labels
5. Split data into training and testing sets
6. Scale features using StandardScaler
7. Train neural network model
8. Evaluate model on test data

---

## Model Performance

Example result:

Test Accuracy: 88.7%
Test Loss: 0.446

This indicates strong classification performance and good generalization capability.

Evaluation methods include:

* Accuracy score
* Classification report
* Confusion matrix
* Training and validation curves


---

## Applications in Digital Forensics

This model can assist forensic investigators by:

* Automatically classifying biometric records
* Supporting identity verification
* Detecting patterns in forensic biometric datasets
* Improving efficiency of forensic analysis

---
