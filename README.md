
<div align="center">

# Lightning Template with Model Compression Techniques

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/DeepVoltaire/AutoAugment.git"><img alt="AutoAugment" src="https://img.shields.io/badge/-AutoAugment-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://timm.fast.ai/"><img alt="timm" src="https://img.shields.io/badge/-timm-4682B4?style=flat&logo=github&labelColor=gray"></a>

<a href="https://github.com/VainF/Torch-Pruning/tree/master"><img alt="Torch-Pruning" src="https://img.shields.io/badge/-TorchPruning-FFC0CB?style=flat&logo=github&labelColor=gray"></a>

</div>

# Overview

This repository contains a PyTorch-Lightning template designed to streamline the process of conducting deep learning experiments. Alongside typical features, this also contains compression methods to reduce the size and speed up your deep learning models.



# Installation

```bash
git clone https://github.com/kiaraesguerra/lightning-template
cd lightning-template
git clone https://github.com/kiaraesguerra/AutoAugment
conda create -n myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```

# Features
* Selection of models, datasets, initialization methods, metrics, and hyperparameters
* Tasks:
  - [X] classification
  - [ ] semantic segmentation
  - [ ] object detection
* Compression techniques:
  - [X] structural pruning
  - [ ] post-training quantization
  - [ ] quantization-aware training
  - [ ] low rank factorization
* QoL features:
  - [ ] generate report containing metrics
  - [ ] comparison of model size and inference time of model before and after compression


# Training


### 1. Training a model with pretrained weights
```
python main.py --model resnet18 --pretrained --dataset cifar10 --prune --method group_norm --speed-up 2 --autoaugment 
```

### 2. Training a model from scratch
```
python main.py --model resnet18 --weight-init 'DeltaInit' --dataset cifar10  --speed-up 2 --autoaugment 
```

### 3. Training a model you've previously trained
```
python main.py --model resnet18 --pretrained-path 'path/to/pretrained_model.pt' --dataset cifar10 --speed-up 2 --autoaugment 
```


# Pruning

Pruning can be applied to a model with pretrained weights, from scratch, or from a model that you have previously trained. However, it is not advised to implement the technique from scratch since this will lead to terrible model performance. 

### 1. Pruning a pretrained model
```
python main.py --model resnet18 --pretrained --dataset cifar10 --callbacks prune --method group_norm --speed-up 2 --autoaugment
```

### 2. Pruning a model you've previously trained
This method is especially useful in pruning techniques which have an option for sparsity learning such as 'group_sl' and 'slim'. Sparsity learning is an additional training step which regularizes the model to make it more suitable for pruning. The procedure is as follows: Pretrain -> Sparsity Learning -> Prune -> Finetune

```
python main.py --model resnet18 --pretrained-path 'path/to/pretrained_model.pt' --dataset cifar10 --callbacks prune --method group_norm --speed-up 2 --autoaugment 
```


