# Neural Language Model Assignment (PyTorch)

## Overview

This project implements a neural language model from scratch in PyTorch, as specified in the assignment guidelines. The model is trained on "Pride and Prejudice" by Jane Austen (Project Gutenberg), demonstrating sequence learning, model configuration experiments, and text generation.

## Dataset

- Source: `Pride_and_Prejudice-Jane_Austen.txt` [Project Gutenberg][file:2]
- Automatically cleaned and split into train, validation, and test sets.


## Setup Instructions

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

```


### Step 2: Prepare Data

Place `Pride_and_Prejudice-Jane_Austen.txt` in the root directory.
Run:

```bash
python prepare_data.py

```


This will output cleaned `train.txt`, `val.txt`, and `test.txt` files to `data/`.

### Step 3: Train Model(s)

```bash
python train.py

```


The script will train three models (underfit, overfit, best-fit) and save results, checkpoints, and plots.

### Step 4: Text Generation (Inference)

Use the best model to generate new text or complete sentences:

```bash
python inference.py

```


## Model Configurations

| Model      | Embedding | Hidden Units | Layers | Dropout | Learning Rate | Epochs | Batch Size | Intended Behavior          |
|------------|-----------|--------------|--------|---------|--------------|--------|------------|---------------------------|
| Underfit   | 64        | 128          | 1      | 0.2     | 0.01         | 15     | 128        | Demonstrates underfitting  |
| Overfit    | 512       | 1024         | 3      | 0.1     | 0.0005       | 50     | 32         | Demonstrates overfitting   |
| Best Fit   | 256       | 512          | 2      | 0.5     | 0.001        | 30     | 64         | Optimal generalization     |


## Key Features

- PyTorch implementation (LSTM-based)
- Vocabulary building from the dataset
- Customizable sequence architecture (RNN/GRU/LSTM/Transformer possible)
- Perplexity evaluation
- Model checkpointing and early stopping
- Gradient clipping & learning rate scheduling
- Text generation with temperature control
- Sentence completion functionality
- Reproducible experiments (fixed random seed)
- Support for both word-level and character-level tokenization

## Hardware & Runtime

- GPU: Recommended (CUDA)
- RAM: At least 8GB for efficient DataLoader usage
- Estimated Training Time: 1-2 hours (GPU), 4+ hours (CPU)

