import torch
import torch.nn as nn
import torch.nn.functional as F

def benchmark_mode(flag):
    torch.backends.cuda.matmul.allow_tf32 = flag
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
    torch.backends.cudnn.allow_tf32 = flag
    torch.backends.cudnn.benchmark = flag
    torch.backends.cudnn.deterministic = not flag
    
class ResNetBasicblock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, path=None, temp=None):
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

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input_size = 1
        self.flops = 1
        for s in module.weight.size():
            self.flops*=s
        self.flops*=output.size(2)*output.size(3)
        for i in input[0].size():
            self.input_size*=i
        module.flops = self.flops
        module.input_size = self.input_size/(16*32*32)
    
    def close(self):
        self.hook.remove()

def profile(net):
    hook = []
    for m in net.modules():
        if isinstance(m, Prunedblock) or isinstance(m, Deformableblock):
            hook.append(Hook(m.conv_a))
            hook.append(Hook(m.conv_b))
    net.eval()
    with torch.no_grad():
        net(torch.randn(1,3,32,32).cuda())
    net.train()
    for h in hook:
        h.close()
        
class Deformableblock(ResNetBasicblock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, path=None, temp=None):
        super(Deformableblock, self).__init__(inplanes, planes, stride, downsample)
        self.n_path = path
        self.temp = temp
        self.order = 0.5
        self.group_a = nn.Parameter(torch.zeros(path, planes))
        self.group_b = nn.Parameter(torch.zeros(path, planes))

    def set_arch(self, hard=False):
        if hard:
            index_a = self.group_a.max(dim=0, keepdim=True)[1]
            self.prob_a = torch.zeros_like(self.group_a).scatter_(0, index_a, 1.0)
            index_b = self.group_b.max(dim=0, keepdim=True)[1]
            self.prob_b = torch.zeros_like(self.group_b).scatter_(0, index_b, 1.0)
        else:
            self.prob_a = F.gumbel_softmax(self.group_a/self.temp, dim=0)
            self.prob_b = F.gumbel_softmax(self.group_b/self.temp, dim=0)

    @torch.no_grad()
    def calc_penalty(self, mode=0):
        self.penalty_a = torch.zeros_like(self.conv_a.weight)
        self.penalty_b = torch.zeros_like(self.conv_b.weight)
        if mode==1:
            self.pgrad_a = torch.zeros_like(self.prob_a)
            self.pgrad_b = torch.zeros_like(self.prob_b)
            self.buffer_a = []
            self.buffer_b = []
                
        for p, pg in zip(self.prob_a, self.pgrad_a):
            penalty, grad = self.derivatives(self.conv_a.weight, p, self.buffer_a, self.order, mode)
            scale = (self.conv_a.input_size**0.5)*self.conv_a.weight.size(2)*((self.group_a.size(0)/self.conv_a.weight.size(0))**1.5)
            self.penalty_a.add_(scale*penalty)
            if mode==2:
                pg.add_(grad)
            
        for p, pg in zip(self.prob_b, self.pgrad_b):
            penalty, grad = self.derivatives(self.conv_b.weight, p, self.buffer_b, self.order, mode)
            scale = (self.conv_b.input_size**0.5)*self.conv_b.weight.size(2)*((self.group_b.size(0)/self.conv_b.weight.size(0))**1.5)
            self.penalty_b.add_(scale*penalty)
            if mode==2:
                pg.add_(grad)

    @torch.no_grad()
    def derivatives(self, w, p, buffer, order, mode):
        group = p.view(-1,1,1,1)*w
        group_norm = (group**2).sum(dim=(3,2,0),keepdim=True)**0.5
        multiplier = group/(1e-8+group_norm)
        g_order = p**order
        g_order_sum = g_order.sum()
        gnorm = g_order_sum**(1/order)
        
        penalty= gnorm*p.view(-1,1,1,1)*multiplier
        if mode==2:
            grad = gnorm*(multiplier*w).sum(dim=(3,2,1)) + group_norm.sum()*gnorm*g_order/(g_order_sum*p+1e-8)
            return penalty, grad
        elif mode==1:
            grad = gnorm*multiplier+gnorm*p.view(-1,1,1,1)*(1/(group_norm+1e-8)-group.sum(dim=(3,2,0),keepdim=True)/(group_norm**2+1e-8)*multiplier)*w,\
                   (multiplier*p.view(-1,1,1,1)),\
                    gnorm*g_order/(g_order_sum*p+1e-8)
            buffer.append(grad)
            return penalty, None
        else:
            return penalty, None
   
    @torch.no_grad()         
    def calc_group_grad(self):
        grad_a = self.conv_a.weight - self.checkpoint_a
        grad_b = self.conv_b.weight - self.checkpoint_b
        
        for pg, g in zip(self.pgrad_a, self.buffer_a):
            g1, g2, g3 = g
            grad = (grad_a*g1).sum(dim=(3,2,1))+(grad_a*g2).sum()*g3
            pg.add_(grad)
        for pg, g in zip(self.pgrad_b, self.buffer_b):
            g1, g2, g3 = g
            grad = (grad_b*g1).sum(dim=(3,2,1))+(grad_b*g2).sum()*g3
            pg.add_(grad)
        
        
    def group_backward(self, rate):
        (rate*((self.prob_a*self.pgrad_a).sum()+(self.prob_b*self.pgrad_b).sum())).backward()
    
    @torch.no_grad()
    def do_penalty(self, rate, lr):
        self.conv_a.weight.add_(self.penalty_a, alpha=-rate*lr)
        self.conv_b.weight.add_(self.penalty_b, alpha=-rate*lr)
            
    @torch.no_grad()
    def checkpoint(self):
        self.checkpoint_a = self.conv_a.weight.clone().detach()
        self.checkpoint_b = self.conv_b.weight.clone().detach()
    
class Prunedblock(ResNetBasicblock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, path=None, temp=None):
        super(Prunedblock, self).__init__(inplanes, planes, stride, downsample)
        self.group_a = nn.Parameter(torch.zeros(path, planes))
        self.group_b = nn.Parameter(torch.zeros(path, planes))
        self.register_buffer('mask_conv_a', torch.ones(planes, inplanes, 1, 1))
        self.register_buffer('mask_conv_b', torch.ones(planes, planes, 1, 1))
        self.register_buffer('mask_bn_a', torch.ones(planes))
        self.register_buffer('mask_bn_b', torch.ones(planes))
    
    @torch.no_grad()    
    def pruning_stats(self, verbose=False):
        flops = self.mask_conv_a.sum().item()/self.mask_conv_a.numel(), self.mask_conv_b.sum().item()/self.mask_conv_b.numel()
        if verbose:
            print(flops)
            
        return (flops[0], self.conv_a.flops, self.conv_a.weight.numel()), (flops[1] , self.conv_b.flops, self.conv_b.weight.numel())

    @torch.no_grad()
    def set_arch(self):
            index_a = self.group_a.max(dim=0, keepdim=True)[1]
            self.prob_a = torch.zeros_like(self.group_a).scatter_(0, index_a, 1.0)
            index_b = self.group_b.max(dim=0, keepdim=True)[1]
            self.prob_b = torch.zeros_like(self.group_b).scatter_(0, index_b, 1.0)
    
    @torch.no_grad()
    def set_mask(self, rate):
        mask_ai, mask_ao, stats_a = self.find_mask(self.conv_a.weight, self.prob_a, rate, 0)
        mask_bi, mask_bo, stats_b = self.find_mask(self.conv_b.weight, self.prob_b, rate, rate)
                
        self.mask_bn_b.copy_(mask_bo)
        self.mask_conv_b.copy_(mask_bo.view(-1,1,1,1)*mask_ao.view(1,-1,1,1)*mask_bi.view(*mask_bi.size(),1,1))
        self.mask_bn_a.copy_((self.mask_conv_b.sum(0)>0).float().squeeze())
        self.mask_conv_a.copy_(self.mask_bn_a.view(-1,1,1,1)*mask_ai.view(*mask_ai.size(),1,1))
        
        print(list(zip(stats_a,(self.prob_a*self.mask_bn_a.unsqueeze(0)).sum(1).long().tolist())),
             list(zip(stats_b,(self.prob_b*self.mask_bn_b.unsqueeze(0)).sum(1).long().tolist())))
            
    @torch.no_grad()
    def find_mask(self, weight, prob, rate_i, rate_o):
        importance = weight.data**2
            
        imp_i = torch.stack([((p.view(-1,1,1,1)**2)*importance).sum(dim=(3,2,0)) for p in prob],dim=0)
        imp_o = importance.sum(dim=(3,2,1))

        imp_i = imp_i/(imp_i.sum(dim=1,keepdim=True)+1e-12)
        imp_o = imp_o/(imp_o.sum()+1e-12)

        rank_i = imp_i.sort(dim=1)[0]
        rank_o = imp_o.sort(dim=0)[0]

        csoi_i = rank_i.cumsum(dim=1)
        csoi_o = rank_o.cumsum(dim=0)

        count_i = (csoi_i<rate_i).long().sum(dim=1)
        count_o = (csoi_o<rate_o).long().sum(dim=0)
            
        th_i = rank_i[torch.arange(rank_i.size(0)), count_i-1].unsqueeze(1)
        th_o = rank_o[count_o-1]

        mask_i = (prob.unsqueeze(2) * (imp_i > th_i).float().unsqueeze(1)).sum(0)
        
        if count_o!=0:
            mask_o = (imp_o > th_o).float()
        else:
            mask_o = torch.ones_like(imp_o)
        
        pruned_stats = (imp_i > th_i).float().sum(1).long().tolist()
        
        return mask_i, mask_o, pruned_stats
        
    @torch.no_grad()
    def prune(self):
        self.conv_a.weight.mul_(self.mask_conv_a)
        self.bn_a.weight.mul_(self.mask_bn_a)
        self.bn_a.bias.mul_(self.mask_bn_a)
        self.conv_b.weight.mul_(self.mask_conv_b)
        self.bn_b.weight.mul_(self.mask_bn_b)
        self.bn_b.bias.mul_(self.mask_bn_b)

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

from typing import List

class Sampler(nn.Module):
    def __init__(self, inds: torch.Tensor):
        super(Sampler, self).__init__()
        self.inds = inds
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.take_along_dim(x, self.inds, 1)

class Splitter(nn.Module):
    def __init__(self):
        super(Splitter, self).__init__()
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return torch.tensor_split(x, x.size(1), 1)

class Composer(nn.Module):
    def __init__(self, inds: List[int]):
        super(Composer, self).__init__()
        self.inds = inds
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([x[n] for n in self.inds], 1)

def index_add_(a, b, inds):
    return a.index_add_(1, inds, b)

class Splitted_Sequential(nn.Sequential):
    def forward(self, input:List[torch.Tensor]):
        for module in self:
            input = module(input)
        return input

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
            mask_conv_a = block.mask_conv_a
            mask_conv_b = block.mask_conv_b
            self.block = block
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
            
            self.layers_a = nn.ModuleList([nn.Sequential(Sampler(torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device)),
                                                         *l,
                                                         Splitter()) for l, inds in zip(layers_a, inp_inds_a)])
            self.layers_b = nn.ModuleList([Splitted_Sequential(Composer(inds),
                                                         *l) for l, inds in zip(layers_b, inp_inds_b)])
            
            
            self.inp_a = [torch.LongTensor(inds).view(1,-1,1,1).to(pbn_b.weight.device) for inds in inp_inds_a]
            self.out_a = out_inds_a
            self.inp_b = inp_inds_b
            self.out_b = [torch.LongTensor(inds).to(pbn_b.weight.device) for inds in out_inds_b]

        self.downsample = block.downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity   
        outs_a = []
        for group_a in self.layers_a:
            outs_a += group_a(x)
        outs_b = [group_b(outs_a) for group_b in self.layers_b]
        
        for i, o in enumerate(outs_b):
            index_add_(out, o, self.out_b[i])
        
        return out.relu_()

class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, path, temp):
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
        self.stage_1 = self._make_layer(block, 16, layer_blocks, path, temp, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, path, temp, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, path, temp, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, path, temp, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, path = path, temp=temp))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,  path = path, temp=temp))

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
