# Dynamic Structure Pruning for Compressing CNNs

_Jun-Hyung Park, Yeachan Kim, Junho Kim, Joon-Young Choi, and SangKeun Lee_

_Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI-23)_

We will release pruned models (pytorch-JIT-compiled) soon!

## Requirements
- Python 3.7
- PyTorch 1.10.0
- TorchVision 0.11.0
- tqdm

## How to use DSP in your code

You should first train pre-trained models to learn groups and then prune and finetune the group-learned models.

Our group-learning and pruning modules require three steps.
1. Defining a wrapper
2. Initializing
3. Processing after every update (step)

Following sections show code examples using our modules

**Differentiable Group Learning**

```
from dsp_module import *

...

# After defining your model, optimizer, criterion, etc.
group_trainer = GroupWrapper(model, optimizer, criterion, regularization_power, total_num_iterations, num_groups, temperature)

...

# Training iteration
for epoch in range(args.epochs):
    for x, y in train_dataloader:
        # Before forward (model(x))
        group_trainer.initialize()
        out = model(x)
        ...

        # After model update (optimizer.step())
        group_trainer.after_step(x, y)

...

```

**Group Channel Pruning**

```
from dsp_module import *

...

# Before loading group-learned checkpoints
pruner = PruneWrapper(model, fp_every_nth_conv, num_groups)

...

# Before training starts
flops, params = pruner.initialize(pruning_rate)

# Training iteration
for epoch in range(args.epochs):
    for x, y in train_dataloader:
        
        ...

        # After model update (optimizer.step())
        pruner.after_step()
...
```

Please refer to our CIFAR-10 pruning codes (cifar_dsp.py and cifar_finetune.py) to help your understanding of our modules.

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
# ResNet20 with group 4, lambda=2e-3
python cifar_dsp.py -l 20 -g 4 -r 2e-3

# ResNet20 with group 2, lambda=2e-3
python cifar_dsp.py -l 20 -g 2 -r 2e-3

# ResNet56 with group 4, lambda=5e-4
python cifar_dsp.py -l 56 -g 4 -r 5e-4
```

**Group Channel Pruning**

```
# ResNet20 with group 4, pruning rate=0.5
python cifar_finetune.py -l 20 -g 4 -p 0.5

# ResNet56 with group 4, pruning rate=0.5
python cifar_finetune.py -l 56 -g 4 -p 0.5

```

**Packing Pruned Models**

```
python pack_model.py --ckpt [pruned_model_path] --save [save_path]
```

| Model           |  ACC  | P.FLOPS | P.PARAMS  | CKPT     |
| --------------- | ----- | ------- | --------- | -------- |
| ResNet20 (g=4)  | 92.19 |  63.83  |   48.98   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet20_g4.pt) |
| ResNet20 (g=3)  | 92.16 |  62.93  |   49.35   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet20_g3.pt) |
| ResNet20 (g=2)  | 91.95 |  62.14  |   47.09   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet20_g2.pt) |
| ResNet56 (g=4)  | 94.34 |  63.42  |   55.29   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet56_g4.pt) |
| ResNet56 (g=3)  | 94.13 |  62.99  |   53.74   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet56_g3.pt) |
| ResNet56 (g=2)  | 94.08 |  61.17  |   53.14   | [Link](https://github.com/irishev/DSP/raw/main/checkpoints/resnet56_g2.pt) |

How to use checkpoints
```
import torch
cnn = torch.load('[CKPT_PATH]')
```

_We slightly changed the implementation of regularization scaling to obtain better speedup._

_As a result, pruned results may be different from those in the paper (usually more pruned FLOPS and fewer pruned parameters)._

## TODO
- Implement model-agnostic pruner