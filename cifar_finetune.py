import os
import math
import copy
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from cifar_model import *
from dsp_module import *
import argparse
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

benchmark_mode(True)

parser = argparse.ArgumentParser(description='CIFAR-10 ResNet Training')
parser.add_argument('--save_dir', type=str, default='./cifarmodel/', help='Folder to save checkpoints and log.')
parser.add_argument('-l', '--layers', default=20, type=int, metavar='L', help='number of ResNet layers')
parser.add_argument('-d', '--device', default='0', type=str, metavar='D', help='main device (default: 0)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='J', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='E', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='B', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.015, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, metavar='W', help='weight decay')

# Fine-tuning Hyperparameters
parser.add_argument('-c', '--cycles', default=5, type=int, metavar='C', help='number of cyclic iterations')
parser.add_argument('-g', '--groups', default=4, type=int, metavar='G', help='number of groups')
parser.add_argument('-p', '--prune', default=0.5, type=float, metavar='P', help='pruning rates)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.device



def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
    
device = torch.device("cuda")

def train(network):
    train_dataset = dsets.CIFAR10(root='./dataset',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                           std=(0.2470, 0.2435, 0.2616))
                                  ]))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, drop_last=True)
    test_dataset = dsets.CIFAR10(root='./dataset',
                               train=False,
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2470, 0.2435, 0.2616))
                               ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size, num_workers=args.workers,
                                              shuffle=False)

    cnn, netname = network
    config = netname
    savepath = args.save_dir+'/'+netname+'_P%sg%dc%.2f.pkl'%(args.device, args.groups, args.prune)
    loadpath = args.save_dir+'/'+netname+'_G%sg%d.pkl'%(args.device, args.groups)
    state_dict, baseacc = torch.load(loadpath)
    print(loadpath)
    print(baseacc)
    pruner = PruneWrapper(cnn, args.groups, 2)
    cnn.load_state_dict(state_dict, strict=False)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    epoch_per_cycle = math.ceil(args.epochs / args.cycles)
    scheduler = CosineAnnealingLR(optimizer, epoch_per_cycle)
    
    flops, params = pruner.initialize(args.prune)
    bestset = {'acc':0, 'flops':flops, 'params':params, 'state_dict': copy.deepcopy(cnn.state_dict())}

    bar = tqdm(total=len(train_loader) * args.epochs, ncols=120)
    for epoch in range(args.epochs):
        cnn.train()
        for step, (images, labels) in enumerate(train_loader):
            
            optimizer.zero_grad()
            gpuimg = images.to(device)
            labels = labels.to(device)

            outputs = cnn(gpuimg)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            pruner.after_step()
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|PR_F:%.2f|PR_P:%.2f" % (get_lr(optimizer)[0], loss.item(), bestset['acc'], bestset['flops'], bestset['params']))
            bar.update()

        scheduler.step()
        cnn.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = cnn(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels).sum().item()
        acc = 100 * correct / total
        print()
        print(f"Val accuracy: {acc}%")
        cnn.train()

        if bestset['acc']<=acc:
            bestset = {'acc':acc, 'flops':flops, 'params':params, 'state_dict': copy.deepcopy(cnn.state_dict())}
            torch.save([bestset['state_dict'], bestset['acc']], savepath)
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|PR_F:%.2f|PR_P:%.2f" % (get_lr(optimizer)[0], loss.item(), bestset['acc'], bestset['flops'], bestset['params']))
        
        # prune a small portion of channels for each cycle
        # to prevent over-fitting and to remove dead channels
        if (epoch<args.epochs-1) and ((epoch+1)%epoch_per_cycle==0):
            cnn.load_state_dict(bestset['state_dict'])
            flops, params=pruner.initialize(bestset['flops']/100+0.001)
            optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, epoch_per_cycle)
            
    bar.close()
    return bestset['acc']

def resnet(layers):
    return CifarResNet(ResNetBasicblock, layers, 10).to(device), "resnet"+str(layers)

if __name__ == '__main__':
    train(resnet(args.layers))
