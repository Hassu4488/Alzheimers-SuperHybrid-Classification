# Alzheimers-SuperHybrid-Classification
Late-fusion deep learning architecture combining Swin Transformers and Tri-CNNs (Resnet50, Inceptionv3, xception) for Alzheimer's disease classification.
# Global-Local Feature Fusion for Alzheimer's Disease Classification

[DOI](https://doi.org/10.5281/zenodo.20451193)

This repository contains the codebase for evaluating various deep learning architectures—including a novel "Super-Hybrid" fusion model—for the multi-class classification of Alzheimer's disease from MRI scans.

## 🧠 Project Overview
Detecting early-stage Alzheimer's requires analyzing both macro-level neurodegeneration (e.g., ventricular enlargement) and micro-level structural anomalies (e.g., localized cortical atrophy). This project implements and evaluates multiple architectures before fusing them into a final model:

1. **Standalone Architectures:** ResNet50, InceptionV3, Xception, Swin Transformer.
2. **Tri-CNN Hybrid:** An ensemble of the three CNNs acting as high-frequency edge detectors.
3. **Super-Hybrid (Late-Fusion):** Synthesizes global context (Swin Transformer) with local precision (Tri-CNN) via a custom dense classification head.

## 📁 Repository Structure
To maintain readability and manage GPU memory constraints, the models are separated into distinct files:

* `01_Swin_Transformer.ipynb`: Training and evaluation of the standalone Vision Transformer.
* `02_ResNet_Inception_Xception.ipynb`: Training pipelines for the individual CNN backbones.
* `03_Tri_CNN_Ensemble.ipynb`: Feature concatenation and training of the CNN ensemble.
* `04_Super_Hybrid_Model.ipynb`: The final late-fusion architecture and master classification head.

## 🚀 Installation & Usage

### Prerequisites
The code is optimized for Python 3.8+ and PyTorch. It was originally developed in a Kaggle/Google Colab environment.

Because this entire work has been done within a single notebook, it is crucial to execute the data preprocessing steps in the correct order.
```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn opencv-python pytorch-grad-cam
