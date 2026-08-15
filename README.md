# MSCNet-TCN-Transformer

### A Hierarchical Spatiotemporal Network with CBAM Attention and Euclidean Alignment for Motor Imagery EEG Decoding

## MSCNet-TCN-Transformer for Motor Imagery EEG Decoding

**Core idea:** Multi-scale CNN + CBAM attention + TCN + Transformer + Euclidean Alignment

### Abstract

Decoding motor imagery electroencephalogram (MI-EEG) signals is essential for the development of brain-computer interface (BCI) systems. However, reliable MI-EEG decoding remains challenging due to the complex spatiotemporal characteristics of EEG signals and variations across recording sessions.

This work proposes **MSCNet-TCN-Transformer**, a hierarchical deep learning framework that progressively learns multi-scale, temporal, and global representations from MI-EEG signals. The framework employs a **Multi-Scale Convolutional Network (MSCNet)** with integrated **CBAM attention** to extract and refine local spatiotemporal representations at multiple temporal resolutions. A **Temporal Convolutional Module (TCM)** subsequently fuses and compresses these multi-scale features, while a shallow **Transformer encoder** captures global temporal dependencies. The TCM and Transformer representations are combined through a dual-stream connection for final MI-EEG classification.

To improve robustness to session-level variations and limited training data, **Euclidean Alignment (EA)**, segment-mixing augmentation, MixUp regularization, label smoothing, and warm-up cosine annealing with AdamW are incorporated into the training pipeline.

The proposed framework is evaluated on the **BCI Competition IV-2a and IV-2b datasets** under a subject-dependent classification setting, achieving average accuracies of **83.91%** and **88.63%**, respectively.

---

## Overall Framework

The proposed MSCNet-TCN-Transformer progressively extracts multi-scale spatiotemporal features and refines them through temporal convolution and Transformer-based global attention.

<img width="1437" height="963" alt="image" src="https://github.com/user-attachments/assets/da4102d4-7010-403d-ae09-48c0d70490e8" />

---

## Dataset & Preprocessing

The proposed framework is evaluated on the **BCI Competition IV Dataset 2a and Dataset 2b** for motor imagery EEG classification.

### BCI Competition IV-2a

- **Subjects:** 9
- **EEG channels:** 22
- **Sampling rate:** 250 Hz
- **Motor imagery classes:** 4
  - Left hand
  - Right hand
  - Both feet
  - Tongue

### BCI Competition IV-2b

- **Subjects:** 9
- **EEG channels:** 3
- **Sampling rate:** 250 Hz
- **Motor imagery classes:** 2
  - Left hand
  - Right hand

The original dataset files are **not included in this repository**. They should be obtained from the official BCI Competition IV source and used according to the applicable dataset terms and citation requirements.

**Dataset:** https://www.bbci.de/competition/iv/

### Preprocessing

The EEG preprocessing pipeline consists of:

- 50 Hz notch filtering
- Motor imagery trial extraction
- Per-trial, per-channel normalization
- Euclidean Alignment (EA)
- Global normalization using training-set statistics

---

## Experimental Setup

The proposed **MSCNet-TCN-Transformer** is evaluated under a **subject-dependent classification** setting on the BCI Competition IV-2a and IV-2b datasets.

The model is trained using:

- **Optimizer:** AdamW
- **Learning-rate scheduler:** Cosine Annealing
- **Warm-up:** 20 epochs
- **MixUp:** α = 0.2
- **Label smoothing:** 0.05

To improve generalization, the training pipeline incorporates **segment-mixing augmentation** and **MixUp regularization**.

The proposed model is compared with established MI-EEG deep learning approaches, including:

- ShallowConvNet
- DeepConvNet
- EEGNet
- EEGInception
- TSception
- EEGTCNet
- ADFCNN
- TCANet

Performance is evaluated using **classification accuracy** and **Cohen's Kappa**.

---

## Results

The proposed **MSCNet-TCN-Transformer** achieves an average accuracy of **83.91%** on BCI Competition IV-2a and **88.63%** on BCI Competition IV-2b.

### BCI Competition IV-2a

| Model | Accuracy (%) | Cohen's Kappa |
|---|---:|---:|
| ShallowConvNet | 73.46 | 0.6461 |
| DeepConvNet | 76.43 | 0.6857 |
| EEGNet | 75.38 | 0.6718 |
| EEGInception | 66.63 | 0.5550 |
| TSception | 58.72 | 0.4496 |
| EEGTCNet | 76.08 | 0.6811 |
| ADFCNN | 78.32 | 0.7109 |
| TCANet | 83.06 | 0.7742 |
| **MSCNet-TCN-Transformer (Ours)** | **83.91** | **0.7963** |

### BCI Competition IV-2b

| Model | Accuracy (%) | Cohen's Kappa |
|---|---:|---:|
| ShallowConvNet | 84.64 | 0.6929 |
| DeepConvNet | 85.88 | 0.7176 |
| EEGNet | 86.44 | 0.7288 |
| EEGInception | 87.69 | 0.7538 |
| TSception | 80.99 | 0.6197 |
| EEGTCNet | 85.80 | 0.7160 |
| ADFCNN | 85.87 | 0.7175 |
| TCANet | 88.52 | 0.7703 |
| **MSCNet-TCN-Transformer (Ours)** | **88.63** | **0.8485** |

---

## Citation

If you use this work, please cite the original BCI Competition IV datasets and the related TCANet work.

### Related Work

Zhao, W. et al.  
*TCANet: Temporal Convolutional Attention Network for MI-EEG Decoding.*  
Cognitive Neurodynamics, 2025.

### Dataset

This work uses the **BCI Competition IV Dataset 2a and Dataset 2b**. Please follow the dataset's original citation and usage requirements when using the data.

---

## Acknowledgements

We acknowledge the creators and providers of the BCI Competition IV datasets and the authors of the related research that forms the basis for this work.
