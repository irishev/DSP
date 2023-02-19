# Dynamic Structure Pruning for Compressing CNNs

_Jun-Hyung Park, Yeachan Kim, Junho Kim, Joon-Young Choi, and SangKeun Lee_

_Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI-23)_

We will release pruned models (pytorch-JIT-compiled) soon!

## Requirements
- Python 3.7
- PyTorch 1.10.0
- TorchVision 0.11.0
- tqdm

## Pruning on CIFAR-10 

| Model           |  ACC  | P.FLOPS | P.PARAMS  | CKPT     |
| --------------- | ----- | ------- | --------- | -------- |
| ResNet20 (g=4)  | 92.16 |  68.01  |   51.78   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet20_g4.pt) |
| ResNet20 (g=3)  | 92.16 |  66.49  |   52.49   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet20_g3.pt) |
| ResNet20 (g=2)  | 92.03 |  64.74  |   52.21   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet20_g2.pt) |
| ResNet56 (g=4)  | 94.13 |  63.83  |   54.21   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet56_g4.pt) |
| ResNet56 (g=3)  | 93.91 |  62.21  |   51.43   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet56_g3.pt) |
| ResNet56 (g=2)  |       |         |           | [Link]() |

How to use checkpoints
```
import torch
cnn = torch.load('[CKPT_PATH]')
```

_We slightly changed the implementation of regularization scaling to obtain better speedup._

_As a result, pruned results may be different from those in the paper (usually more pruned FLOPS and fewer pruned parameters)._

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

**Packing Pruned Models**

```
python pack_model.py --ckpt [pruned_model_path] --save [save_path]
```
