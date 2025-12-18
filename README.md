# LLM_Time_Series

This repository contains an ECG foundation model for ECG analysis.

## Repository Structure

```
.
├── preprocess/           # Data preprocessing scripts and notebooks
│   ├── preprocess_PTB.py
│   ├── preprocess_CS.py
│   ├── preprocess_CPSC2018.py
│   └── preprocess_MIMIC.ipynb
├── src/                  # Core logic and helper functions
│   ├── Autoencoder.py    # Waveform autoencoder for latent representation
│   ├── xresnet_embed.py  # ECG feature extraction using XResNet
│   ├── kmeans_gpu.py     # GPU-accelerated clustering for tokenization
│   └── models/           # Model architecture definitions
│       ├── xresnet1d.py
│       └── basic_conv1d.py
├── tokenizer.ipynb       # Notebook for tokenization and clustering analysis
├── pretrain.py           # Main pre-training script
├── finetune.py           # Main fine-tuning script
└── requirement.txt       # Project dependencies
```

## Setup

1.  **Environment**: Create a Python environment.
2.  **Dependencies**: Install the required packages:
    ```bash
    pip install -r requirement.txt
    ```

## Usage

### 1. Preprocessing
Run the preprocessing scripts to prepare the ECG segments from different datasets (MIMIC-IV, PTB-XL, etc.).
Example:
```bash
python preprocess/preprocess_PTB.py
```

### 2. Tokenization
To discrete the time series to tokens, using tokenizer.ipynb

### 3. Pre-training
The pre-training phase involves training the BERT model on tokenized ECG sequences.
```bash
torchrun --nproc_per_node=8 pretrain.py
```

### 4. Fine-tuning
After pre-training, fine-tune the model on specific downstream tasks:
```bash
python finetune_xresnet_final.py --dataset_name ptb --num_classes 5
```
