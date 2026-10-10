# Grayscale Vegetable Classification with CNNs

> **School project.** Coursework for ST1504 Deep Learning (CA1, Part A), Diploma in Applied AI & Analytics, Singapore Polytechnic. Kept here as a record of my early deep learning work.

Classifies grayscale vegetable images into 11 classes using CNNs built from scratch in Keras. Transfer learning was not allowed.

## What's in the notebook

- EDA: class balance, pixel intensity histograms, per-class average images, t-SNE of the flattened images
- Two input resolutions (23×23 and 101×101) to compare how resolution affects learning
- Three architectures (baseline, normalised, enhanced) trained with different optimisers, augmentation levels and learning-rate schedulers
- Class weights to handle the imbalanced training set
- Model selection on the average of validation accuracy, test accuracy, macro F1 and weighted F1

Data: 7,699 training, 2,200 validation and 2,200 test images.

## Result

| Resolution | Selected model |
|---|---|
| 23×23 | Enhanced CNN, baseline augmentation, Adam, ReduceLROnPlateau |
| 101×101 | Enhanced CNN, aggressive augmentation, Nadam, no scheduler |

## Files

- `CNN Model Notebook.ipynb`: the full notebook with outputs
- `CNN Model Notebook.html`: the same notebook exported to HTML for reading in a browser

Stack: Python, TensorFlow/Keras, scikit-learn, OpenCV, pandas, seaborn.
