# LLM_Time_Series

This repository contains an ECG foundation model for ECG analysis.

## Repository Structure

```
.
├── preprocess/
│   ├── preprocess_PTB.py
│   ├── preprocess_CS.py
│   ├── preprocess_CPSC2018.py
│   └── preprocess_MIMIC.ipynb
├── src/
│   ├── Autoencoder.py
│   ├── xresnet_embed.py
│   ├── kmeans_gpu.py
│   └── models/
│       ├── xresnet1d.py
│       └── basic_conv1d.py
├── tokenizer.ipynb
├── pretrain.py
├── finetune.py
└── requirement.txt
```

## Usage

### 1. Preprocessing
Preprocess the ECG data from different datasets.
Example:
```bash
python preprocess/preprocess_PTB.py
```

### 2. Tokenization
To discretize the time series into tokens, using tokenizer.ipynb

### 3. Pre-training
The pre-training phase involves training the BERT model on tokenized ECG sequences.
```bash
torchrun --nproc_per_node=8 pretrain.py
```

### 4. Fine-tuning
After pre-training, fine-tune the model on downstream tasks:
```bash
python finetune.py --dataset_name ptb --num_classes 5
```
