import os, math
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from cifar_model import *
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
parser.add_argument('--epochs', default=300, type=int, metavar='E', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='B', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.015, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, metavar='W', help='weight decay')

# Fine-tuning Hyperparameters
parser.add_argument('-c', '--cycles', default=5, type=int, metavar='C', help='number of cyclic iterations')
parser.add_argument('-g', '--groups', default=4, type=int, metavar='G', help='number of groups')
parser.add_argument('-p', '--prune', default=0.15, type=float, metavar='P', help='pruning rates)')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.device


def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]

device = torch.device("cuda")

def train(network, pr_rate, path):
    train_dataset = dsets.CIFAR10(root='./dataset',
                                  train=True,
                                  download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, 4),
                                      #transforms.TrivialAugmentWide(interpolation=transforms.InterpolationMode.BILINEAR),
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
    savepath = args.save_dir+'/'+netname+'_P%sg%dc%.2f.pkl'%(args.device, path, pr_rate)
    loadpath = args.save_dir+'/'+netname+'_G%sg%d.pkl'%(args.device, path)
        
    state_dict, baseacc = torch.load(loadpath)
    print(savepath, baseacc)
    cnn.load_state_dict(state_dict, strict=False)

    param_optimizer = list(cnn.named_parameters())
    exclude = ['group']
    optimizer_grouped_parameters = [p for n, p in param_optimizer if not any(nd in n for nd in exclude)]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for m in cnn.modules():
        if isinstance(m, Prunedblock):
            m.set_arch()
            m.set_mask(pr_rate)
            m.prune()
    
    profile(cnn)
    
    remaining = 0
    total = 0
    params = 0
    rparams = 0
    for l in [m.pruning_stats(True) for m in cnn.modules() if isinstance(m, Prunedblock)]:
        for rate, flop, param in l:
            remaining += rate*flop
            total += flop
            rparams += rate*param
            params += param
    
    flops = (1-remaining/total)*100
    params = (1-rparams/params)*100
    print('FLOP pruning rate:', flops, '\nParam pruning rate:', params)
    bestset = {'acc':0, 'flops':flops, 'params':params}

    epoch_per_cycle = math.ceil(args.epochs / args.cycles)
    scheduler = OneCycleLR(optimizer, args.lr, epoch_per_cycle)
        
    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        scheduler.step(epoch%epoch_per_cycle)
        loss = train_epoch(train_loader, cnn, config, criterion, bestset, optimizer, bar)
        acc = evaluate(test_loader, cnn)

        if bestset['acc']<=acc:
            bestset = {'acc':acc, 'flops':flops, 'params':params}
            torch.save([cnn.state_dict(),bestset['acc']], savepath)
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|PR_F:%.2f|PR_P:%.2f" % (get_lr(optimizer)[0], loss.item(), bestset['acc'], bestset['flops'], bestset['params']))
        
        # prune a small portion of channels for each cycle
        # to prevent over-fitting and to remove dead channels
        if (epoch<args.epochs-1) and ((epoch+1)%epoch_per_cycle==0):
            for m in cnn.modules():
                if isinstance(m, Prunedblock):
                    m.set_mask(1e-3)
                    m.prune()
            remaining = 0
            total = 0
            params = 0
            rparams = 0
            for l in [m.pruning_stats(True) for m in cnn.modules() if isinstance(m, Prunedblock)]:
                for rate, flop, param in l:
                    remaining += rate*flop
                    total += flop
                    rparams += rate*param
                    params += param
            flops = (1-remaining/total)*100
            params = (1-rparams/params)*100
            print('FLOP pruning rate:', flops, '\nParam pruning rate:', params)
            
    bar.close()
    return bestset['acc']

def evaluate(test_loader, cnn):
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
    print(acc)
    cnn.train()
    return acc

def train_epoch(train_loader, cnn, config, criterion, bestset, optimizer, bar):
    for step, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        gpuimg = images.to(device)
        labels = labels.to(device)

        outputs = cnn(gpuimg)
        loss = criterion(outputs, labels)
            
        loss.backward()
        optimizer.step()
        prune(cnn)
        
        bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|PR_F:%.2f|PR_P:%.2f" % (get_lr(optimizer)[0], loss.item(), bestset['acc'], bestset['flops'], bestset['params']))
        bar.update()
    return loss

def prune(cnn):
    for m in cnn.modules():
        if isinstance(m, Prunedblock):
            m.prune()

def resnet(layers, path):
    return CifarResNet(Prunedblock, layers, 10, path=path, temp=None).to(device), "resnet"+str(layers)

if __name__ == '__main__':
    train(resnet(args.layers, args.groups), args.prune, args.groups)