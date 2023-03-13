import torch
import torch.nn as nn
import torch.nn.functional as F

def benchmark_mode(flag):
    torch.backends.cudnn.benchmark = flag
    torch.backends.cudnn.deterministic = not flag
    
class ResNetBasicblock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = basicblock.relu_()

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        out = residual + basicblock

        return out.relu_()

from typing import Tuple, List

class Sampler_T1(nn.Module):
    def __init__(self, ind1: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return torch.take_along_dim(x, self.ind1, 1),
    
class Sampler_T2(nn.Module):
    def __init__(self, ind1: torch.Tensor, ind2: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
        self.ind2 = ind2
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return torch.take_along_dim(x, self.ind1, 1), torch.take_along_dim(x, self.ind2, 1)

class Sampler_T3(nn.Module):
    def __init__(self, ind1: torch.Tensor, ind2: torch.Tensor, ind3: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return torch.take_along_dim(x, self.ind1, 1), torch.take_along_dim(x, self.ind2, 1), torch.take_along_dim(x, self.ind3, 1)

class Sampler_T4(nn.Module):
    def __init__(self, ind1: torch.Tensor, ind2: torch.Tensor, ind3: torch.Tensor, ind4: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3
        self.ind4 = ind4
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return torch.take_along_dim(x, self.ind1, 1), torch.take_along_dim(x, self.ind2, 1), torch.take_along_dim(x, self.ind3, 1), torch.take_along_dim(x, self.ind4, 1)

class Concat_T1(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[0]
    
class Concat_T2(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat((x[0], x[1]), dim=1)
     
class Concat_T3(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat((x[0], x[1], x[2]), dim=1)
    
class Concat_T4(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat((x[0], x[1], x[2], x[3]), dim=1)
      
class IAdd_T1(nn.Module):
    def __init__(self, ind1: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
    
    def forward(self, out: torch.Tensor, x: List[torch.Tensor]) -> torch.Tensor:
        out = out.index_add_(1, self.ind1, x[0])
        return out
    
class IAdd_T2(nn.Module):
    def __init__(self, ind1: torch.Tensor, ind2: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
        self.ind2 = ind2
    
    def forward(self, out: torch.Tensor, x: List[torch.Tensor]) -> torch.Tensor:
        out = out.index_add_(1, self.ind1, x[0])
        out = out.index_add_(1, self.ind2, x[1])
        return out

class IAdd_T3(nn.Module):
    def __init__(self, ind1: torch.Tensor, ind2: torch.Tensor, ind3: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3
    
    def forward(self, out: torch.Tensor, x: List[torch.Tensor]) -> torch.Tensor:
        out = out.index_add_(1, self.ind1, x[0])
        out = out.index_add_(1, self.ind2, x[1])
        out = out.index_add_(1, self.ind3, x[2])
        return out
    
class IAdd_T4(nn.Module):
    def __init__(self, ind1: torch.Tensor, ind2: torch.Tensor, ind3: torch.Tensor, ind4: torch.Tensor):
        super().__init__()
        self.ind1 = ind1
        self.ind2 = ind2
        self.ind3 = ind3
        self.ind4 = ind4
    
    def forward(self, out: torch.Tensor, x: List[torch.Tensor]) -> torch.Tensor:
        out = out.index_add_(1, self.ind1, x[0])
        out = out.index_add_(1, self.ind2, x[1])
        out = out.index_add_(1, self.ind3, x[2])
        out = out.index_add_(1, self.ind4, x[3])
        return out
    
class Compactblock(nn.Module):
    expansion=1
    def __init__(self, *args, **kwargs):
        super(Compactblock, self).__init__()
        
    def compact(self, block):
        with torch.no_grad():
            pconv_a = block.conv_a
            pbn_a = block.bn_a
            pconv_b = block.conv_b
            pbn_b = block.bn_b
            mask_conv_a = block.conv_a.mask
            mask_conv_b = block.conv_b.mask
            stride = block.conv_a.weight.size(0)//block.conv_a.weight.size(1)
            
            inp_inds_a = []
            out_inds_a = []
            inp_inds_b = []
            out_inds_b = []
            ch_map1 = []
            m_a = mask_conv_a.view(*mask_conv_a.size()[:2])
            m_b = mask_conv_b.view(*mask_conv_b.size()[:2])
            m_a_dict = {}
            m_b_dict = {}
            for n, i in enumerate(m_a):
                id = ''.join([str(k) for k in (i==1).long().tolist()])
                if i.sum()==0:
                    continue
                if id not in m_a_dict:
                    in_ind = i.nonzero().view(-1).tolist()
                    m_a_dict[id] = [in_ind,[]]
                m_a_dict[id][1].append(n)
            
            
            for n, i in enumerate(m_b):
                id = ''.join([str(k) for k in (i==1).long().tolist()])
                if i.sum()==0:
                    continue
                if id not in m_b_dict:
                    in_ind = i.nonzero().view(-1).tolist()
                    m_b_dict[id] = [in_ind, []]
                m_b_dict[id][1].append(n)
            
            group_a = list(m_a_dict.values())
            group_b = list(m_b_dict.values())
            
            layer_info = [[], []]
            layers_a = [[] for _ in range(len(group_a))]
            layers_b = [[] for _ in range(len(group_b))]
            pos = 0
            for g, l in zip(group_a, layers_a):
                ich_ind, och_ind = g
                num_in = len(ich_ind)
                num_out = len(och_ind)
                layer_info[0].append((num_in,num_out))
                l.append(nn.Conv2d(num_in, num_out, 3, stride, 1, bias=False).to(pbn_b.weight.device))
                l.append(nn.BatchNorm2d(num_out).to(pbn_b.weight.device))
                l.append(nn.ReLU(inplace=True))
                l[0].weight.copy_(pconv_a.weight[och_ind][:,ich_ind])
                l[1].weight.copy_(pbn_a.weight[och_ind])
                l[1].bias.copy_(pbn_a.bias[och_ind])
                l[1].running_mean.copy_(pbn_a.running_mean[och_ind])
                l[1].running_var.copy_(pbn_a.running_var[och_ind])
                inp_inds_a.append(ich_ind)
                out_inds_a.append(och_ind)
                ch_map = {}
                for k, o in enumerate(och_ind):
                    ch_map[o]=k+pos
                ch_map1.append(ch_map)
                pos += len(ch_map)
                    
            
            for g, l in zip(group_b, layers_b):
                ich_ind, och_ind = g
                mapped_inds = []
                reord = []
                for m in ch_map1:
                    for i in ich_ind:
                        if i in m:
                            mapped_inds.append(m[i])
                            reord.append(i)
                ich_ind = reord
                num_in = len(ich_ind)
                num_out = len(och_ind)
                layer_info[1].append((num_in,num_out))
                l.append(nn.Conv2d(num_in, num_out, 3, 1, 1, bias=False).to(pbn_b.weight.device))
                l.append(nn.BatchNorm2d(num_out).to(pbn_b.weight.device))
                l[0].weight.copy_(pconv_b.weight[och_ind][:,ich_ind])
                l[1].weight.copy_(pbn_b.weight[och_ind])
                l[1].bias.copy_(pbn_b.bias[och_ind])
                l[1].running_mean.copy_(pbn_b.running_mean[och_ind])
                l[1].running_var.copy_(pbn_b.running_var[och_ind])
                
                inp_inds_b.append(mapped_inds)
                out_inds_b.append(och_ind)
            
            self.n_groups_a = len(inp_inds_a)
            self.layers_a = nn.ModuleList([nn.Sequential(*l) for l in layers_a])
            if self.n_groups_a==1:
                self.sampler_a = Sampler_T1(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_a])
                self.concat_a = Concat_T1()
            elif self.n_groups_a==2:
                self.sampler_a = Sampler_T2(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_a])
                self.concat_a = Concat_T2()
            elif self.n_groups_a==3:
                self.sampler_a = Sampler_T3(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_a])
                self.concat_a = Concat_T3()
            elif self.n_groups_a==4:
                self.sampler_a = Sampler_T4(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_a])
                self.concat_a = Concat_T4()
            self.n_groups_b = len(inp_inds_b)
            self.layers_b = nn.ModuleList([nn.Sequential(*l) for l in layers_b])
            if self.n_groups_b==1:
                self.sampler_b = Sampler_T1(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_b])
                self.residual = IAdd_T1(*[torch.LongTensor(inds).to(pbn_b.weight.device) for inds in out_inds_b])
            elif self.n_groups_b==2:
                self.sampler_b = Sampler_T2(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_b])
                self.residual = IAdd_T2(*[torch.LongTensor(inds).to(pbn_b.weight.device) for inds in out_inds_b])
            elif self.n_groups_b==3:
                self.sampler_b = Sampler_T3(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_b])
                self.residual = IAdd_T3(*[torch.LongTensor(inds).to(pbn_b.weight.device) for inds in out_inds_b])
            elif self.n_groups_b==4:
                self.sampler_b = Sampler_T4(*[torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_b])
                self.residual = IAdd_T4(*[torch.LongTensor(inds).to(pbn_b.weight.device) for inds in out_inds_b])
            

        self.downsample = block.downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity
        
        ch_a = self.sampler_a(x)
        out_a = []
        for i, l in enumerate(self.layers_a):
            out_a.append(l(ch_a[i]))
        ch_b = self.sampler_b(self.concat_a(out_a))
        out_b = []
        for i, l in enumerate(self.layers_b):
            out_b.append(l(ch_b[i]))
        
        out = self.residual(out, out_b)
        
        return out.relu_()

class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        x = x.relu_()
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
