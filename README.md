# LLM_Time_Series

This repository contains an ECG foundation model for ECG analysis.

## Usage

### 1. Preprocessing
Preprocess the ECG data from different datasets using preprocess.ipynb

### 2. Tokenization
Discretize the time series into tokens using tokenizer.ipynb

### 3. Pre-training
The pre-training phase involves training the BERT model on tokenized ECG sequences.
```bash
torchrun --nproc_per_node=n pretrain.py
```

### 4. Fine-tuning
Fine-tune the model on downstream tasks:
```bash
python finetune.py --dataset_name name --num_classes 5
```
