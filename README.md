# 🔍 VerifAura — AI-Powered Document Forgery Detection

<p align="center">
  <strong>Detect potential document tampering using Error Level Analysis (ELA) and Deep Convolutional Neural Networks.</strong>
</p>

<p align="center">
  <a href="https://verifaura-ai.onrender.com">🌐 Live Demo</a>
  &nbsp;•&nbsp;
  <a href="https://github.com/aashya-madhan/VerifAura">📂 Source Code</a>
</p>

---

## 📌 Overview

**VerifAura** is an AI-powered document forgery detection system that combines **Error Level Analysis (ELA)** with a lightweight **Convolutional Neural Network (CNN)** to identify potential image manipulation.

The system accepts both individual images and common document formats such as **PDF, DOCX, and PPTX**, processes documents page-by-page, and provides a forgery prediction along with an ELA visualization and confidence score.

The project was designed with a focus on **lightweight inference, explainable visual output, and practical document processing**.

---

## 🎯 Problem

Digital documents and images can be modified using image-editing tools without leaving obvious visual traces.

Traditional visual inspection can make it difficult to identify subtle manipulations such as:

* Image splicing
* Region replacement
* Copy-paste manipulation
* Localized editing
* Other compression inconsistencies

VerifAura attempts to identify these inconsistencies automatically using **ELA-based preprocessing followed by CNN classification**.

---

## 💡 Solution

VerifAura follows a multi-stage processing pipeline:

```text
                  ┌──────────────────┐
                  │  Upload Document │
                  │  or Image        │
                  └────────┬─────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Document Processing  │
                │ PDF / DOCX / PPTX    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Error Level Analysis │
                │        (ELA)         │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Lightweight CNN      │
                │ Classification       │
                └──────────┬───────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │ Prediction + Confidence   │
              │ + ELA Heatmap             │
              └───────────────────────────┘
```

---

## ✨ Key Features

* 🔍 **AI-powered forgery detection**
* 🧠 CNN-based authentic vs. forged classification
* 🖼️ Error Level Analysis (ELA) visualization
* 📄 PDF analysis page-by-page
* 📝 DOCX image extraction and analysis
* 📊 PPTX slide-level analysis
* 📈 Confidence scoring
* ⚡ Lightweight CNN architecture
* 🌐 Flask-based web interface
* 💻 CPU-friendly model training

---

## 🧠 How It Works

### 1. Error Level Analysis

The uploaded image is resaved at a fixed JPEG quality and compared against the original.

The resulting pixel-level differences are amplified to produce an **ELA image**. Regions that have undergone different compression histories may appear with different error levels.

Implementation:

```text
src/ela.py
```

---

### 2. CNN Classification

The generated ELA image is passed to a custom CNN trained to classify the input as:

```text
Authentic
   or
Forged
```

The model uses a lightweight architecture designed to reduce computational and storage requirements.

Implementation:

```text
src/model.py
```

---

### 3. Document Processing

For supported document formats, the system converts or extracts content into analyzable images before applying the same ELA + CNN pipeline.

```text
PDF
 └── Pages → Images → ELA → CNN

DOCX
 └── Embedded Images / Rendered Content → ELA → CNN

PPTX
 └── Slides → Images → ELA → CNN
```

Implementation:

```text
src/document_processor.py
```

---

## 📊 Dataset & Model

The model was trained using the **CASIA v2.0** image forgery dataset.

| Metric               |               Result |
| -------------------- | -------------------: |
| Dataset              |           CASIA v2.0 |
| Images               |              ~12,600 |
| Classification       | Authentic vs. Forged |
| Accuracy             |                 80%+ |
| CNN Parameters       |                 ~50K |
| Training Environment |                  CPU |
| Training Time        |         < 15 minutes |
| Model Size           |      97 MB → 0.37 MB |
| Model Size Reduction |                99.6% |

> **Note:** Accuracy is dataset-dependent and should not be interpreted as a guarantee of real-world forgery detection performance.

---

## 📁 Project Structure

```text
VerifAura/
│
├── models/
│   └── best_model_smoke.keras
│
├── src/
│   ├── app.py
│   ├── predict.py
│   ├── ela.py
│   ├── document_processor.py
│   ├── model.py
│   ├── train.py
│   └── prepare_dataset.py
│
├── static/
│   ├── style.css
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── .gitignore
├── requirements.txt
└── runtime.txt
```

---

## 📄 Supported File Types

| File Type     | Formats                   | Processing                               |
| ------------- | ------------------------- | ---------------------------------------- |
| 🖼️ Image     | PNG, JPG, JPEG, BMP, TIFF | ELA + CNN                                |
| 📑 PDF        | `.pdf`                    | Page rasterization → ELA + CNN           |
| 📝 Word       | `.docx`, `.doc`           | Image extraction / rendering → ELA + CNN |
| 📊 PowerPoint | `.pptx`, `.ppt`           | Slide extraction → ELA + CNN             |

---

## 🛠️ Tech Stack

<div align="center">

| Category                | Technologies                      |
| ----------------------- | --------------------------------- |
| **Language**            | Python                            |
| **Machine Learning**    | TensorFlow, CNN                   |
| **Image Processing**    | Pillow, Error Level Analysis      |
| **Web Framework**       | Flask                             |
| **Document Processing** | PyMuPDF, python-docx, python-pptx |
| **Deployment**          | Gunicorn                          |
| **Dataset**             | CASIA v2.0                        |

</div>

---

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* pip
* Git

### Installation

```bash
git clone https://github.com/aashya-madhan/VerifAura.git

cd VerifAura

python -m venv venv
```

### Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
cd src
python app.py
```

Open:

```text
http://localhost:5000
```

---

## 🔬 Technical Highlights

### Lightweight CNN

The model uses a compact architecture with approximately **50K parameters**, helping reduce computational and storage requirements.

### Global Average Pooling

GlobalAveragePooling is used to reduce the number of trainable parameters while retaining spatial feature information.

### Regularization

Dropout and data augmentation are used to improve generalization and reduce overfitting.

### Early Stopping

Training uses early stopping to avoid unnecessary epochs once validation performance stops improving.

### Model Compression

The deployed model was reduced from approximately **97 MB to 0.37 MB**, resulting in a reported **99.6% reduction in model size**.

---

## 📸 Demo

### Upload

Upload an image or supported document through the web interface.

### Analysis

VerifAura processes the document and generates ELA representations for analysis.

### Result

The system returns:

* Authentic / Forged prediction
* Confidence score
* ELA visualization
* Page-level results for supported documents

> **Tip:** Add 2–3 screenshots of the actual application here. A screenshot of the upload page and result page would significantly improve the repository presentation.

---

## ⚠️ Limitations

VerifAura is an experimental machine-learning project and should not be treated as a definitive forensic verification tool.

Performance can vary depending on:

* Image compression
* Dataset distribution
* Manipulation technique
* Document quality
* Previously processed images
* Differences between training and real-world data

Predictions should therefore be considered **indicators of potential manipulation rather than proof of forgery**.

---

## 🔮 Future Improvements

* [ ] Improve performance on unseen manipulation techniques
* [ ] Expand training data beyond CASIA v2.0
* [ ] Add additional forensic features
* [ ] Improve localization of manipulated regions
* [ ] Add batch document processing
* [ ] Add detailed downloadable analysis reports
* [ ] Improve model evaluation with precision, recall, F1-score and confusion matrix
* [ ] Add automated testing and CI/CD

---

## 📌 Project Highlights

```text
~12,600 training images
80%+ reported classification accuracy
~50K CNN parameters
99.6% model-size reduction
PDF / DOCX / PPTX support
CPU-friendly training
```

---

