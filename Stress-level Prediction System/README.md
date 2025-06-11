# Emotion Recognition using Image Processing and Machine Learning

This project focuses on building a robust emotion recognition system using image processing and machine learning techniques. The model is capable of accurately classifying human emotions from facial expressions using a dataset of 35,686 images sourced from Kaggle.

##  Project Overview

The goal of this project is to develop an end-to-end pipeline that processes facial images and predicts emotional states. It combines classical machine learning algorithms with image preprocessing techniques to extract meaningful features for classification.

### Objectives:
- Develop an image processing pipeline to prepare facial images for analysis.
- Train and compare multiple machine learning models for emotion classification.
- Evaluate model performance on a large-scale, labeled facial emotion dataset.

##  Technologies Used

- **Programming Language**: Python  
- **Libraries**: OpenCV, TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib  
- **Algorithms**: Logistic Regression (LR), k-Nearest Neighbors (KNN), Naive Bayes (NB)  
- **Dataset**: [Kaggle - Facial Expression Recognition Dataset]

##  Model Development

- **Preprocessing**: Images were resized, normalized, and converted to grayscale using OpenCV.  
- **Feature Extraction**: Pixel intensity and edge-based features were extracted to train the models.  
- **Model Training**: LR, KNN, and NB models were trained and validated using stratified k-fold cross-validation.  
- **Evaluation**: Model performance was assessed using accuracy, precision, recall, and confusion matrices.

##  Outcomes

- Built a reliable image-based emotion classification model using machine learning.  
- Identified the best-performing model based on balanced accuracy and robustness.  
- Demonstrated potential for applications in human-computer interaction, mental health monitoring, and social robotics.
