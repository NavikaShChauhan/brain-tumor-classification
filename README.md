# 🧠 Brain Tumor Classification using MRI Images
### NeuroLite-Net — A Novel Lightweight Deep Learning Architecture

---

## 📌 Overview

This project proposes **NeuroLite-Net**, a novel lightweight convolutional neural network designed specifically for brain MRI tumor classification. The model classifies MRI scans into four categories without relying on pretrained ImageNet weights, achieving **98.50% test accuracy** with only **0.59M parameters**.

### Classes

| Class | Description |
|-------|-------------|
| Glioma | Malignant tumor arising from glial cells |
| Meningioma | Tumor arising from the meninges |
| Pituitary | Tumor located in the pituitary gland region |
| No Tumor | Healthy brain scan with no tumor |

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.50%** |
| Macro F1-Score | **98.59%** |
| Weighted F1-Score | **98.50%** |
| Mean AUC (all classes) | **0.9982** |
| Best Validation Accuracy | **98.90%** |
| Parameters | **592,296 (~0.59M)** |
| Model Size | **2.26 MB** |
| No Tumor Recall | **100.00%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.9814 | 0.9732 | 0.9773 |
| Meningioma | 0.9939 | 0.9818 | 0.9878 |
| No Tumor | 0.9868 | 1.0000 | 0.9934 |
| Pituitary | 0.9798 | 0.9907 | 0.9852 |

### ROC-AUC Scores

| Class | AUC |
|-------|-----|
| Glioma | 0.9964 |
| Meningioma | 0.9994 |
| No Tumor | 0.9992 |
| Pituitary | 0.9976 |

---

## 🔬 Proposed Architecture — NeuroLite-Net

NeuroLite-Net is built on four custom-designed blocks:

### Block 1 — Multi-Scale Depthwise Separable (MSDS) Block
Parallel 3×3 and 5×5 depthwise separable convolution branches operating simultaneously. Captures both fine-grained textures and broader structural tumor patterns in a single block.

### Block 2 — Residual Micro-Block (RMB)
Lightweight residual connections using depthwise separable convolutions instead of standard convolutions. Preserves gradient flow while reducing parameters by 8–9× compared to standard ResNet blocks.

### Block 3 — Squeeze-and-Excitation Attention (SEA) Module
Channel-wise attention mechanism that suppresses irrelevant background channels (skull, CSF) and amplifies tumor-relevant feature channels.

### Block 4 — Adaptive Feature Fusion (AFF) Module
Fuses shallow (low-level texture) and deep (high-level semantic) features via learnable 1×1 projections, creating an internal feature pyramid without FPN overhead.

### Architecture Flow

```
Input (3×224×224)
    ↓
Stem Conv (32, 3×3, stride=2)          →  32×112×112
MSDS Block 1 (48 filters) + SEA        →  96×112×112
MaxPool                                →  96×56×56
RMB Block 1 (96 filters)               →  96×56×56
MSDS Block 2 (96 filters) + SEA        → 192×56×56
MaxPool                                → 192×28×28  ← AFF shallow
RMB Block 2 (192 filters)              → 192×28×28
MSDS Block 3 (128 filters) + SEA       → 256×28×28
MaxPool                                → 256×14×14
RMB Block 3 (256 filters)              → 256×14×14
MSDS Block 4 (128 filters) + SEA       → 256×14×14
AFF Module (shallow + deep fusion)     → 256×14×14
Global Average Pooling                 → 256
Dropout (0.4) → Linear (128) → Dropout (0.3) → Linear (4)
```

---

## 📊 Comparative Analysis

| Model | Params (M) | Size (MB) | Pretrained | Test Acc |
|-------|-----------|-----------|------------|----------|
| **NeuroLite-Net (Ours)** | **0.59** | **2.26** | **❌ From Scratch** | **98.50%** |
| MobileNetV2 | 2.39 | 13.7 | ✅ ImageNet | — |
| EfficientNet-B0 | 4.17 | 20.4 | ✅ ImageNet | — |
| ShuffleNetV2 | 1.4 | 5.5 | — | — |
| SqueezeNet | 1.2 | 4.7 | — | — |

NeuroLite-Net achieves competitive accuracy with 7× fewer parameters than EfficientNet-B0, trained entirely from scratch without ImageNet pretraining.

---

## 🗂️ Dataset

The dataset is a merged collection from three public sources totaling **13,351 MRI images**:

| Source | Link |
|--------|------|
| Figshare Brain Tumor Dataset | [Link](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| Kaggle — Sartaj Bhuvaji | [Link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) |
| Kaggle — Masoud Nickparvar | [Link](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |

### Class Distribution

| Class | Images |
|-------|--------|
| Glioma | ~3,264 |
| Meningioma | ~3,267 |
| Pituitary | ~3,434 |
| No Tumor | ~3,386 |
| **Total** | **13,351** |

### Folder Structure
```
BrainTumorMRI/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

> ⚠️ Dataset not included due to size. Download from the links above and place in `BrainTumorMRI/` folder.

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/NavikaShChauhan/brain-tumor-classification.git
cd brain-tumor-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download from the Kaggle links above and place images in:
```
brain-tumor-classification/
└── BrainTumorMRI/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Run the Notebook
```bash
jupyter notebook Brain_Tumor_Classification_PyTorch_CLEAN.ipynb
```

Run all cells from top to bottom. Training takes approximately **1–2 hours** on a 6GB GPU.

---

## 📂 Project Structure

```
brain-tumor-classification/
│
├── Brain_Tumor_Classification_PyTorch_CLEAN.ipynb  ← Main notebook
├── requirements.txt                                 ← Python dependencies
├── README.md                                        ← This file
├── .gitignore                                       ← Git ignore rules
│
└── BrainTumorMRI/                                   ← Dataset (not uploaded)
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

---

## 🛠️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 224 × 224 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| LR Schedule | Warm-up (5 epochs) + Cosine Annealing |
| Loss Function | CrossEntropyLoss (label smoothing=0.1) |
| Epochs | 60 (max) |
| Early Stopping | patience=12 |
| Gradient Clipping | max_norm=1.0 |
| Mixed Precision | AMP (float16) |
| Train/Val/Test Split | 70% / 15% / 15% |

---

## 🔍 Explainable AI

Grad-CAM, Grad-CAM++, and Saliency Maps are used to visualize model decisions:

- **Glioma** — activation focused on tumor mass center
- **Meningioma** — activation on brain periphery where meningiomas grow
- **No Tumor** — diffuse activation with no focal point
- **Pituitary** — activation precisely at sella turcica (pituitary gland location)

The model attends to clinically correct anatomical regions, validating its medical relevance.

---

## 💻 Hardware

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB VRAM) |
| RAM | 24 GB |
| Storage | NVMe SSD |
| OS | Windows 11 |
| CUDA | 12.x |
| Python | 3.10 |
| PyTorch | 2.0+ |

---

## 📦 Dependencies

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
pandas >= 1.5.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scikit-learn >= 1.2.0
opencv-python >= 4.7.0
Pillow >= 9.5.0
torchinfo >= 1.8.0
psutil >= 5.9.0
```

---

## 👤 Author

**Navika Chauhan**
- GitHub: [@NavikaShChauhan](https://github.com/NavikaShChauhan)
- Email: chauhannavika164@gmail.com
