# Dataset

## BCI Competition IV Datasets

This project uses the **BCI Competition IV Dataset 2a** and **Dataset 2b** for motor imagery EEG (MI-EEG) classification.

### Dataset 2a

- 9 subjects
- 22 EEG channels
- 250 Hz sampling frequency
- 4 motor imagery classes:
  - Left hand
  - Right hand
  - Both feet
  - Tongue

### Dataset 2b

- 9 subjects
- 3 EEG channels
- 250 Hz sampling frequency
- 2 motor imagery classes:
  - Left hand
  - Right hand

### Dataset Source

The original datasets were provided as part of the **BCI Competition IV** and are available through the official BCI Competition website:

https://www.bbci.de/competition/iv/

The original dataset files are **not redistributed in this repository**. Users should obtain the datasets directly from the official source and follow the applicable terms of use and citation requirements.

### Usage in This Project

The datasets are used for training and evaluating the proposed EEG classification framework. The downloaded data are processed using the preprocessing pipeline provided in this repository before being used for model training and evaluation.

### Related Work

The proposed framework is based on and extends the **TCANet** architecture introduced by:

> Zhao, W., et al. *TCANet: Temporal Convolutional Attention Network for MI-EEG Decoding*. Cognitive Neurodynamics, 2025.

Users of this repository should appropriately acknowledge and cite the original BCI Competition IV datasets and the related TCANet work when applicable.
