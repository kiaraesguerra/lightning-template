
<hr>
<div align="center">

# Lightning Template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://github.com/DeepVoltaire/AutoAugment.git"><img alt="AutoAugment" src="https://img.shields.io/badge/-AutoAugment-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://timm.fast.ai/"><img alt="timm" src="https://img.shields.io/badge/-timm-4682B4?style=flat&logo=github&labelColor=gray"></a>


</div>

# Overview



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
* Datasets: MNIST, SVHN, CIFAR-10, CIFAR-100,  <a href="https://paperswithcode.com/dataset/cinic-10">CINIC-10</a>
* Models: Plain MLP and CNN
* Initialization methods: kaiming-normal, <a href="https://arxiv.org/pdf/1806.05393.pdf"> delta-orthogonal initialization</a>
* Compression techniques (TO BE ADDED): structural pruning, post-training quantization, quantization-aware training, low rank factorization

# Training

### 1. 

```
python main.py --model resnet18 --weight-init 'delta' --lr 1e-2 --min-lr 0 --scheduler 'cosine' --autoaugment 
```

