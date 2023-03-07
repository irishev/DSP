import os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from cifar_model import *
from dsp_module import *
import argparse
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

benchmark_mode(False)

parser = argparse.ArgumentParser(description='CIFAR-10 ResNet Training')
parser.add_argument('--ckpt', type=str, help='Path to the fine-tuned checkpoint')
parser.add_argument('--save', type=str, help='Path to save a densified model')
parser.add_argument('-l', '--layers', default=20, type=int, metavar='L', help='number of ResNet layers')
parser.add_argument('-g', '--groups', default=4, type=int, metavar='G', help='number of groups')
args = parser.parse_args()

device = torch.device("cuda")

def main():
    test_dataset = dsets.CIFAR10(root='./dataset',
                               train=False,
                               transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                         std=(0.2470, 0.2435, 0.2616))
                               ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128, num_workers=4,
                                              shuffle=False)

    cnn = CifarResNet(ResNetBasicblock, args.layers, 10).to(device)
    pruner = PruneWrapper(cnn, 2, args.groups)
    loadpath = args.ckpt
    state_dict, baseacc = torch.load(loadpath)
    print(args.ckpt, baseacc)
    cnn.load_state_dict(state_dict, strict=False)
    pruner.summary(True, True)
        
    packed_cnn = CifarResNet(Compactblock, args.layers, 10).to(device)

    org_modules = dict(cnn.named_modules())
    new_modules = dict(packed_cnn.named_modules())
    for k in org_modules:
        if isinstance(org_modules[k], ResNetBasicblock):
            new_modules[k].compact(org_modules[k])
    copy_params(cnn, packed_cnn)
    
    for m in packed_cnn.modules():
        if isinstance(m, Compactblock):
            del m.block
            
    input_tensor = torch.randn(1,3,32,32).to(device)
    traced_cnn = torch.jit.trace(packed_cnn.eval(), input_tensor)
    #print(traced_cnn)
    
    evaluate(test_loader, traced_cnn)
    torch.jit.save(traced_cnn, args.save)
        
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

def copy_params(model, inf_model):
    with torch.no_grad():
        inf_model.conv_1_3x3 =model.conv_1_3x3 
        inf_model.bn_1=model.bn_1
        inf_model.classifier=model.classifier
        
main()