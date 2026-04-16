# CNN for Alzheimer's MRI Classification

Binary classification of MRI images into **Healthy** vs. **Mild Dementia** using a Convolutional Neural Network built with TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-orange) ![Accuracy](https://img.shields.io/badge/Test%20Accuracy-89%25-green)

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 89.01% |
| AUC-ROC | 0.96 |
| Precision (Healthy) | 0.91 |
| Recall (Healthy) | 0.87 |
| F1-Score (Healthy) | 0.89 |
| Precision (Mild Dementia) | 0.87 |
| Recall (Mild Dementia) | 0.91 |
| F1-Score (Mild Dementia) | 0.89 |

### Training curves

| Accuracy | Loss |
|---|---|
| ![Accuracy](Accuracy.png) | ![Loss](loss.png) |

### Evaluation

| Confusion Matrix | ROC Curve |
|---|---|
| ![Confusion Matrix](cnf_matrix.png) | ![ROC](auc.png) |

![Metrics](eval_metrics.png)

---

## Model Architecture

- **Conv2D + MaxPooling2D** — 64 filters, 3×3 kernel, ReLU activation; 2×2 pooling to reduce spatial dimensions while retaining key features
- **Flatten** — converts 2D feature maps to a 1D vector for the dense layer
- **Dropout** — prevents overfitting by randomly zeroing neuron values during training
- **Dense layers** — fully connected layers ending in a single sigmoid neuron for binary output
- **Early stopping** — monitors `val_loss` with `patience=3`; restores best weights automatically

**Compiled with:** Binary Cross-Entropy loss · Adam optimizer · Accuracy metric

---

## Dataset

MRI images organized into two classes: `Healthy (0)` and `Mild Dementia (1)`. TensorFlow's `image_dataset_from_directory` handled:
- Automatic label inference from folder structure
- Grayscale conversion (single color channel)
- Image resizing and batching
- Shuffling for training set; fixed order for validation/test

---

## Files

| File | Description |
|---|---|
| `CNN_Model.ipynb` | Full model training, evaluation, and visualization notebook |
| `Technical_Report.docx` | Detailed writeup of architecture decisions and results |
| `Accuracy.png` | Training vs. validation accuracy curve |
| `loss.png` | Training vs. validation loss curve |
| `cnf_matrix.png` | Test set confusion matrix |
| `auc.png` | ROC curve |
| `eval_metrics.png` | Precision, recall, and F1-score by class |
