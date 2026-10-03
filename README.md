# 🎗️ UltraIDetect App

A multi-modal machine learning web application built with **Streamlit** and **PyTorch** designed for 3-class breast cancer detection.

## 🚀 Features
* **Multi-class Classification:** Predicts across 3 distinct diagnostic classes.
* **Interactive Web Interface:** User-friendly forms for feature selection and data input.
* **Fast Inference:** Optimized PyTorch model serving real-time predictions.

## 🛠️ Repository Structure
* `app.py` - Core Streamlit application script and UI layouts.
* `breast_cancer_multimodal_3class.pth` - Pre-trained model weights (stored securely via Git LFS).
* `feature_columns.txt` - Definitions for input features and tabular structures.
* `requirements.txt` - Dependencies and environmental configuration packages.

## 💻 Local Installation

To run this application locally, clone this repository and install the dependencies:

```bash
# Clone the repository (Ensure Git LFS is installed)
git clone https://github.com
cd ultraidetect-app

# Install package dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```
