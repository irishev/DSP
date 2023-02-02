# Dynamic Structure Pruning for Compressing CNNs

_Jun-Hyung Park, Yeachan Kim, Junho Kim, Joon-Young Choi, and SangKeun Lee_

_Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI-23)_

## Requirements
- Python 3.6
- PyTorch 1.10.0
- TorchVision 0.11.0
- tqdm

## Pruning on CIFAR-10 

**Pretraining**

```
# pretrain ResNet20
python cifar_pretrain.py -l 20 [--save-dir ./cifarmodel] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]

# pretrain ResNet56
python cifar_pretrain.py -l 56 [--save-dir ./cifarmodel] [--epochs 164] [--batch-size 128] [--lr 0.1] [--momentum 0.9] [--wd 1e-4]
```

**Differentiable Group Learning**

```
# ResNet20 with group 4, lambda=1e-3
python cifar_dsp.py -l 20 -g 4 -r 1e-3

# ResNet20 with group 2, lambda=1e-3
python cifar_dsp.py -l 20 -g 2 -r 1e-3

# ResNet56 with group 4, lambda=5e-4
python cifar_dsp.py -l 56 -g 4 -r 5e-4
```

**Group Channel Pruning**

```
# ResNet20 with group 4, pruning rate=0.2
python cifar_finetune.py -l 20 -g 4 -p 0.2

# ResNet56 with group 4, pruning rate=0.2
python cifar_finetune.py -l 56 -g 4 -p 0.2

```



