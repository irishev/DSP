import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

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
        
class GroupWrapper(nn.Module):
    def __init__(self, model, optimizer, criterion, reg, total_steps, n_groups=None, temp=None):
        super(GroupWrapper, self).__init__()
        print("Initializing...")
        self.model = model
        self.optimizer = optimizer
        self.optimizer2 = type(optimizer)(self.model.parameters(), lr=self.optimizer.defaults['lr'])
        self.criterion = criterion
        
        self.layers = []
        group_parameters = []
        exclude = ['downsample']
        l = -1
        print("Finding layers to be pruned")
        print("="*80)
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) and all(e not in name for e in exclude):
                if l==-1:
                    l+=1
                    continue
                layer.register_parameter('group', nn.Parameter(torch.zeros(n_groups, layer.weight.size(0), device=layer.weight.device)))
                group_parameters.append(layer.group)
                self.layers.append(layer)
                w_dim = layer.weight.size()
                print(f"[{l}] {name}: {w_dim[0]} filters, {w_dim[1]} channels, {w_dim[2]}x{w_dim[3]} kernels")
                l+=1
                
        print("="*80)
        self.n_groups = n_groups
        self.temp = temp
        self.order = 0.5
        self.reg = reg
        self.steps = 0
        self.total_steps = total_steps
        self.group_optimizer = torch.optim.Adam(group_parameters, lr=0.001, eps=1e-12)
        
        print(f"Number of groups: {self.n_groups}")
        print(f"Regularization coefficient: {self.reg}")
        print(f"Temparature: {self.temp}")
        print(f"Total steps: {self.total_steps}")
        print("="*80)
        hook = []
        for layer in self.layers:
            hook.append(Hook(layer))
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1,3,32,32, device=layer.weight.device))
        self.model.train()
        for h in hook:
            h.close()
            
    def set_arch_hard(self, layer):
        index = layer.group.max(dim=0, keepdim=True)[1]
        layer.prob = torch.zeros_like(layer.group).scatter_(0, index, 1.0)
            
    def set_arch(self, layer):
        layer.prob = F.gumbel_softmax(layer.group/self.temp, dim=0)
        layer.pgrad = torch.zeros_like(layer.prob)
        layer.buffer = []
        
    @torch.no_grad()
    def calc_penalty_no_grad(self, layer):
        layer.penalty = torch.zeros_like(layer.weight)
        for p, pg in zip(layer.prob, layer.pgrad):
            group = p.view(-1,1,1,1)*layer.weight
            group_norm = (group**2).sum(dim=(3,2,0),keepdim=True)**0.5
            multiplier = group/(1e-8+group_norm)
            g_order = p**self.order
            g_order_sum = g_order.sum()
            gnorm = g_order_sum**(1/self.order)
            penalty= gnorm*p.view(-1,1,1,1)*multiplier
            scale = (layer.input_size**0.5)*layer.weight.size(2)*((layer.group.size(0)/layer.weight.size(0))**1.5)
            layer.penalty.add_(scale*penalty)
            
    @torch.no_grad()
    def calc_penalty_first_grad(self, layer):
        layer.penalty = torch.zeros_like(layer.weight)
        for p, pg in zip(layer.prob, layer.pgrad):
            group = p.view(-1,1,1,1)*layer.weight
            group_norm = (group**2).sum(dim=(3,2,0),keepdim=True)**0.5
            multiplier = group/(1e-8+group_norm)
            g_order = p**self.order
            g_order_sum = g_order.sum()
            gnorm = g_order_sum**(1/self.order)
            penalty= gnorm*p.view(-1,1,1,1)*multiplier
            grad = gnorm*multiplier+gnorm*p.view(-1,1,1,1)*(1/(group_norm+1e-8)-group.sum(dim=(3,2,0),keepdim=True)/(group_norm**2+1e-8)*multiplier)*layer.weight,\
                (multiplier*p.view(-1,1,1,1)),\
                gnorm*g_order/(g_order_sum*p+1e-8)
            layer.buffer.append(grad)
            
            scale = (layer.input_size**0.5)*layer.weight.size(2)*((layer.group.size(0)/layer.weight.size(0))**1.5)
            layer.penalty.add_(scale*penalty)
                
    @torch.no_grad()
    def calc_penalty_second_grad(self, layer):
        layer.penalty = torch.zeros_like(layer.weight)
        for p, pg in zip(layer.prob, layer.pgrad):
            group = p.view(-1,1,1,1)*layer.weight
            group_norm = (group**2).sum(dim=(3,2,0),keepdim=True)**0.5
            multiplier = group/(1e-8+group_norm)
            g_order = p**self.order
            g_order_sum = g_order.sum()
            gnorm = g_order_sum**(1/self.order)
            penalty= gnorm*p.view(-1,1,1,1)*multiplier
            grad = gnorm*(multiplier*layer.weight).sum(dim=(3,2,1)) + group_norm.sum()*gnorm*g_order/(g_order_sum*p+1e-8)
            
            scale = (layer.input_size**0.5)*layer.weight.size(2)*((layer.group.size(0)/layer.weight.size(0))**1.5)
            layer.penalty.add_(scale*penalty)
            pg.add_(grad)
        
    @torch.no_grad()         
    def second_order_grad(self, layer):
        grad_a = layer.weight - layer.checkpoint
        for pg, g in zip(layer.pgrad, layer.buffer):
            g1, g2, g3 = g
            grad = (grad_a*g1).sum(dim=(3,2,1))+(grad_a*g2).sum()*g3
            pg.add_(grad)       
        
    def group_backward(self, layer):
        (layer.prob*layer.pgrad).sum().backward()
    
    @torch.no_grad()
    def do_penalty(self, layer):
        layer.weight.add_(layer.penalty, alpha=-self.reg*self.optimizer.param_groups[0]['lr']*(self.steps+1)/self.total_steps)
        
    @torch.no_grad()
    def checkpoint(self, layer):
        layer.checkpoint = layer.weight.clone().detach()
    
    @torch.no_grad()
    def stats(self):
        std = [layer.group.std().item() for layer in self.layers]
        return sum(std)/len(std)
        
    def zero_grad(self):
        self.optimizer.zero_grad(True)
        self.group_optimizer.zero_grad(True)
    
    def apply(self, func, inputs):
        return list(map(func, inputs))
    
    def initialize(self):
        self.apply(self.set_arch, self.layers)
        
    def after_step(self, x, y, amp=False, scaler=None):
        self.apply(self.calc_penalty_first_grad, self.layers)
        self.zero_grad()
        states = copy.deepcopy(self.model.state_dict())
        self.optimizer2.load_state_dict(self.optimizer.state_dict())
        self.apply(self.do_penalty, self.layers)
        self.apply(self.checkpoint, self.layers)
        if amp:
            with torch.cuda.amp.autocast():
                out = self.model(x)
                loss = self.criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer2)
            scaler.update()
        else:
            out = self.model(x)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer2.step()
        self.apply(self.calc_penalty_second_grad, self.layers)
        self.apply(self.do_penalty, self.layers)
        self.apply(self.second_order_grad, self.layers)
        self.zero_grad()
        self.model.load_state_dict(states)
        self.apply(self.group_backward, self.layers)
        self.group_optimizer.step()
        self.apply(self.set_arch_hard, self.layers)
        self.apply(self.calc_penalty_no_grad, self.layers)
        self.apply(self.do_penalty, self.layers)
        self.steps+=1
    
    def forward(self, x):
        return self.model(x)

class PruneWrapper(nn.Module):
    def __init__(self, model, fp_every_nth_conv=None, n_groups=None):
        super(PruneWrapper, self).__init__()
        print("Initializing...")
        self.model = model
        self.layers = []
        self.fp_layers = []
        exclude = ['downsample']
        
        l = -1
        print("Finding layers to be pruned")
        print("="*80)
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) and all(e not in name for e in exclude):
                if l==-1:
                    l+=1
                    continue
                layer.register_parameter('group', nn.Parameter(torch.zeros(n_groups, layer.weight.size(0), device=layer.weight.device)))
                layer.register_buffer('mask', torch.ones(layer.weight.size(0), layer.weight.size(1), 1, 1, device=layer.weight.device))
                self.layers.append(layer)
                w_dim = layer.weight.size()
                print(f"[{l}] {name}: {w_dim[0]} filters, {w_dim[1]} channels, {w_dim[2]}x{w_dim[3]} kernels")
                l+=1
                
                if l%fp_every_nth_conv==0:
                    self.fp_layers.append(layer)
                
        self.n_groups = n_groups
        self.first = True
        
        hook = []
        for layer in self.layers:
            hook.append(Hook(layer))
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1,3,32,32, device=layer.weight.device))
        self.model.train()
        for h in hook:
            h.close()
    
    @torch.no_grad()
    def set_arch_hard(self, layer):
        index = layer.group.max(dim=0, keepdim=True)[1]
        layer.prob = torch.zeros_like(layer.group).scatter_(0, index, 1.0)
            
    @torch.no_grad()
    def find_mask(self, layer):
        importance = layer.weight.data**2
            
        imp = torch.stack([((p.view(-1,1,1,1)**2)*importance).sum(dim=(3,2,0)) for p in layer.prob],dim=0)
        imp = imp/(imp.sum(dim=1,keepdim=True)+1e-12)
        rank = imp.sort(dim=1)[0]
        csoi = rank.cumsum(dim=1)
        count = (csoi<self.beta).long().sum(dim=1)
        th = rank[torch.arange(rank.size(0)), count-1].unsqueeze(1)
        mask = (layer.prob.unsqueeze(2) * (imp > th).float().unsqueeze(1)).sum(0)
        
        layer.mask.copy_(mask.view(mask.size(0),mask.size(1),1,1))
    
    @torch.no_grad()
    def find_mask_fp(self, layer):
        importance = layer.weight.data**2
            
        imp = importance.sum(dim=(3,2,1))
        imp = imp/(imp.sum()+1e-12)
        rank = imp.sort(dim=0)[0]
        csoi = rank.cumsum(dim=0)
        count = (csoi<self.beta).long().sum(dim=0)
        th = rank[count-1]
        mask = (imp > th).float().unsqueeze(1)
        
        layer.mask.mul_(mask.view(mask.size(0),mask.size(1),1,1))
    
    @torch.no_grad()
    def apply_mask(self, layer):
        layer.weight.mul_(layer.mask)
    
    def apply(self, func, inputs):
        return list(map(func, inputs))
    
    def initialize(self, rate, train_loader, n_iter=10):
        print("="*80)
        print("Finding pruning settings to achieve the target pruning rate")
        print("="*80)
        self.apply(self.set_arch_hard, self.layers)
        checkpoints = copy.deepcopy(self.model.state_dict())
        self.beta = 0.15
        lower, upper = 0, 1.
        for _ in range(n_iter):
            pflops, pparams = self.prune(train_loader)
            if pflops>rate*100:
                temp = self.beta
                self.beta = (self.beta+lower)/2
                upper = temp
            else:
                temp = self.beta
                self.beta = (self.beta+upper)/2
                lower = temp
            self.model.load_state_dict(checkpoints)
        pflops, pparams = self.prune(train_loader, True)
        return pflops, pparams
    
    def prune(self, train_loader, verbose=False):
        self.apply(self.find_mask, self.layers)
        self.apply(self.find_mask_fp, self.fp_layers)
        self.apply(self.apply_mask, self.layers)
        for n, (x, y) in enumerate(train_loader):
            F.cross_entropy(self.model(x.cuda()), y.cuda()).backward()
            if n/len(train_loader)>0.2:
                break
        # remove dead filters by tracking zero gradients
        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.mul_((m.weight.grad.abs().sum(dim=(3,2,1), keepdim=True)>0).float())
                    m.weight.mul_((m.weight.grad.abs().sum(dim=(3,2,0), keepdim=True)>0).float())
                    if hasattr(m, 'mask'):
                        m.mask.mul_((m.weight.grad.abs().sum(dim=(3,2,1), keepdim=True)>0).float())
                        m.mask.mul_((m.weight.grad.abs().sum(dim=(3,2,0), keepdim=True)>0).float())
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.mul_((m.weight.grad.abs()>0).float())
                    m.bias.mul_((m.bias.grad.abs()>0).float())
                elif isinstance(m, nn.Linear):
                    m.weight.mul_((m.weight.grad.abs().sum(dim=1, keepdim=True)>0).float())
                    m.bias.mul_((m.bias.grad.abs()>0).float())
            pflops, pparams=self.summary(verbose)
        self.model.zero_grad(True)
            
        return pflops, pparams
    
    def summary(self, verbose=False, init=False):
        if init:
            self.apply(self.set_arch_hard, self.layers)
        remaining_flops = 0
        remaining_params = 0
        total_flops = 0
        total_params = 0
        for n, layer in enumerate(self.layers):
            kernels = (layer.weight.abs().sum(dim=(3,2))>0).float()
            remaining=torch.mm(layer.prob,kernels)
            r_ch = (remaining>0).float().sum(dim=1)
            r_f = (remaining.sum(1)/(r_ch+1e-8)).round()
            remaining_flops += layer.flops*kernels.sum().item()/kernels.numel()
            remaining_params += layer.weight.numel()*kernels.sum().item()/kernels.numel()
            total_flops += layer.flops
            total_params += layer.weight.numel()
            if verbose:
                print("[%d] FLOPS: %2.2f%%" % (n, 100*kernels.sum().item()/kernels.numel()), "Structure:",*list(zip(r_f.long().tolist(), r_ch.long().tolist())))
        pflops = 100*(1-remaining_flops/total_flops)
        pparams = 100*(1-remaining_params/total_params)
        if verbose:
            print("="*80)
            print("Summary")
            print(f"Beta: {self.beta}")
            print(f"FLOPS: {int(remaining_flops)} ({pflops}% pruned)")
            print(f"PARAMS: {int(remaining_params)} ({pparams}% pruned)")
            print("="*80)
        return pflops, pparams
    
    def after_step(self):
        self.apply(self.apply_mask, self.layers)
    
    def forward(self, x):
        return self.model(x)
