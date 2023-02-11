import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
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
parser.add_argument('--epochs', default=120, type=int, metavar='E', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='B', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, metavar='W', help='weight decay')

# DSP Hyperparameters
parser.add_argument('-g', '--groups', default=4, type=int, metavar='G', help='number of groups')
parser.add_argument('-r', '--regularize', default=2e-3, type=float, metavar='R', help='regularization power')
parser.add_argument('-t', '--temparature', default=0.5, type=float, metavar='T', help='temparature for gumbel softmax')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.device



def get_lr(optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]
    
device = torch.device("cuda")

def train(network, reg, path):
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
    loadpath = args.save_dir+'/'+netname+'.pkl'
    savepath = args.save_dir+'/'+netname+'_G%sg%d.pkl'%(args.device, path)
    state_dict, baseacc = torch.load(loadpath)
    print(loadpath)
    print(baseacc)
    cnn.load_state_dict(state_dict, strict=False)
    criterion = nn.CrossEntropyLoss()
    bestacc=0

    param_optimizer = list(cnn.named_parameters())
    
    exclude = ['group']
    optimizer_grouped_parameters = [p for n, p in param_optimizer if not any(nd in n for nd in exclude)]
    group_parameters = [p for n, p in param_optimizer if 'group' in n]

    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    group_optimizer = torch.optim.Adam(group_parameters, lr=0.001, eps=1e-12)
    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    
    profile(cnn)
    
    bar = tqdm(total=len(train_loader) * args.epochs)
    for epoch in range(args.epochs):
        cnn.train()
        scheduler.step(epoch)
        loss = train_epoch(train_loader, cnn, config, criterion, bestacc, optimizer, group_optimizer, bar, reg, ((epoch+1)/args.epochs))
        acc = evaluate(test_loader, cnn)

        if (bestacc<acc) and (epoch>8):
            bestacc=acc
            torch.save([cnn.state_dict(),bestacc], savepath)
            bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f" % (get_lr(optimizer)[0], loss.item(), bestacc))
    bar.close()
    return bestacc

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

def train_epoch(train_loader, cnn, config, criterion, bestacc, optimizer, group_optimizer, bar, reg, warmup):
    train_loader2 = iter(train_loader)
    for step, (images, labels) in enumerate(train_loader):
        gpuimg = images.to(device)
        labels = labels.to(device)
        images2, labels2 = next(train_loader2)
        gpuimg2 = images2.to(device)
        labels2 = labels2.to(device)

        loss = train_step(cnn, criterion, optimizer, group_optimizer, labels, gpuimg, labels2, gpuimg2, reg, warmup)

        bar.set_description("[" + config + "]LR:%.4f|LOSS:%.2f|ACC:%.2f|REG:%.3fe-3" % (get_lr(optimizer)[0], loss.item(), bestacc, reg*warmup*1000))
        bar.update()
    return loss

def zero_grad(optimizer, group_optimizer):
    optimizer.zero_grad(True)
    group_optimizer.zero_grad(True)


def train_step(cnn, criterion, optimizer, group_optimizer, labels, gpuimg, labels2, gpuimg2, reg, warmup):
    zero_grad(optimizer, group_optimizer)
    set_arch(cnn, False)
    outputs = cnn(gpuimg)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    calc_penalty(cnn, 1)
    zero_grad(optimizer, group_optimizer)
    states = save_states(cnn, optimizer)
    regularize(cnn, optimizer, reg)
    checkpoint(cnn)
    outputs = cnn(gpuimg2)
    loss = criterion(outputs, labels2)
    loss.backward()
    optimizer.step()
    calc_penalty(cnn, 2)
    regularize(cnn, optimizer, reg)
    group_grad(cnn)
    group_backward(cnn, reg)
    group_optimizer.step()
    zero_grad(optimizer, group_optimizer)
    load_states(cnn, optimizer, states)
    set_arch(cnn, True)
    calc_penalty(cnn, 0)
    regularize(cnn, optimizer, reg*warmup)
    return loss

def group_grad(cnn):
    for m in cnn.modules():
        if is_pruned(m):
            m.calc_group_grad()

def group_backward(cnn, rate):
    for m in cnn.modules():
        if is_pruned(m):
            m.group_backward(rate)

def regularize(cnn, optimizer, rate):
    for m in cnn.modules():
        if is_pruned(m):
            m.do_penalty(rate, optimizer.param_groups[0]['lr'])

def checkpoint(cnn):
    for m in cnn.modules():
        if is_pruned(m):
            m.checkpoint()

def save_states(cnn, optimizer):
    model_states = {k: v.clone().detach() for k, v in cnn.state_dict().items() if 'group' not in k}
    momenta = [{k: v.clone().detach() for k, v in l.items()} for l in optimizer.state_dict()['state'].values()]
    return model_states, momenta
            
def load_states(cnn, optimizer, states):
    cnn.load_state_dict(states[0], False)
    for l, s in zip(optimizer.state_dict()['state'].values(), states[1]):
        for k in l:
            l[k].copy_(s[k])
            
def set_arch(cnn, hard):
    for m in cnn.modules():
        if is_pruned(m):
            m.set_arch(hard)

def calc_penalty(cnn, mode):
    for m in cnn.modules():
        if is_pruned(m):
            m.calc_penalty(mode)
            
def is_pruned(m):
    return isinstance(m, Deformableblock)

def resnet(layers, path, temp):
    return CifarResNet(Deformableblock, layers, 10, path=path, temp=temp).to(device), "resnet"+str(layers)

if __name__ == '__main__':
    train(resnet(args.layers, args.groups, args.temparature), args.regularize, args.groups)