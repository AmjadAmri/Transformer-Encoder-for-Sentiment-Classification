# Transformer Encoder for Sentiment Classification

A Deep Learning project that implements a Transformer encoder from scratch using PyTorch and applies it to binary sentiment classification on the Rotten Tomatoes Movie Reviews dataset.

The project includes custom implementations of scaled dot-product attention, multi-head attention, Transformer encoder blocks, training experiments, ablation studies, and error analysis.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformer-Encoder-6B46C1" alt="Transformer Encoder">
  <img src="https://img.shields.io/badge/NLP-Sentiment_Classification-0EA5E9" alt="NLP">
  <img src="https://img.shields.io/badge/Ablation_Study-Experimental_Analysis-14B8A6" alt="Ablation Study">
</p>


## Overview

This project focuses on building and evaluating a Transformer encoder model for sentiment classification. Unlike using a ready-made Transformer module, the core components were implemented manually, including scaled dot-product attention, multi-head attention, positional encoding, and Transformer encoder blocks.

The model was trained on the Rotten Tomatoes Movie Reviews dataset to classify reviews as either positive or negative. The project also includes baseline experiments, ablation studies on attention heads and dropout rate, and confusion-matrix-based error analysis.

## Dataset

Dataset: Rotten Tomatoes Movie Reviews

Task: Binary sentiment classification

Target Labels:

- 0: Negative review
- 1: Positive review

Dataset size:

- 10,662 movie reviews
- 5,331 positive reviews
- 5,331 negative reviews

Data split:

- Training: 80% — 8,530 samples
- Validation: 10% — 1,066 samples
- Testing: 10% — 1,066 samples

## Project Workflow

| Stage | Description |
|------|-------------|
| Core Implementation | Implemented scaled dot-product attention, multi-head attention, and Transformer encoder blocks |
| Unit Testing | Verified tensor shapes, mask behavior, attention weights, and gradient flow |
| Data Preprocessing | Cleaned text, tokenized reviews, built vocabulary from training data only, and padded/truncated sequences |
| Model Training | Trained Transformer-based sentiment classifier using PyTorch |
| Baseline Experiments | Compared multiple baseline configurations |
| Ablation Study | Tested the effect of changing attention heads and dropout rate |
| Evaluation | Evaluated the model using validation/test accuracy, loss curves, and confusion matrix |
| Error Analysis | Analyzed misclassified examples and common failure patterns |

## Results

Best baseline results:

| Metric | Value |
|-------|-------|
| Best Validation Accuracy | 0.669 |
| Best Validation Loss | 0.615 |
| Test Accuracy | 0.705 |
| Test Loss | 0.575 |
| Best Epoch | 6 |

## Ablation Study

### Attention Heads

| Model | Validation Accuracy | Test Accuracy |
|------|---------------------|---------------|
| 8 Heads Baseline | 0.669 | 0.705 |
| 4 Heads | 0.679 | 0.721 |
| 2 Heads | 0.685 | 0.709 |

The ablation results showed that reducing the number of attention heads slightly improved validation accuracy, but the differences were small and not sufficient to make a definitive conclusion.

### Dropout Rate

| Model | Validation Accuracy | Test Accuracy |
|------|---------------------|---------------|
| Dropout 0.1 Baseline | 0.669 | 0.705 |
| Dropout 0.3 | 0.611 | 0.627 |

Increasing dropout to 0.3 caused underfitting and reduced overall performance.

## Error Analysis

The error analysis showed that the model struggled with:

- Sarcastic or ironic reviews
- Reviews with mixed positive and negative language
- Short texts with limited context
- Phrases where sentiment depends on subtle semantic interpretation

These errors suggest that a shallow Transformer encoder may not always capture deeper contextual or compositional sentiment patterns.

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Accuracy, Loss Curves, Confusion Matrix |
| Environment | Google Colab / Jupyter Notebook |


```md
## Future Improvements

- Increase Transformer depth by adding more encoder layers.
- Compare learned positional encoding with sinusoidal positional encoding.
- Use subword tokenization instead of word-level tokenization.
- Add macro-F1 and precision/recall metrics.
