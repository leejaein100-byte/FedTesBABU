import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
#from torchvision.models import wide_resnet28_2, WideResNet28_2_Weights
#from .utils import init_param, make_batchnorm, loss_fn
#from config import cfg


class DecConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels, kernel_size, stride, padding, bias):
        super(DecConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.sigma_weight = nn.Parameter(copy.deepcopy(self.weight.data) / 2)
        self.phi_weight = nn.Parameter(copy.deepcopy(self.weight.data) / 2)
        self.weight = None
        if bias:
            self.sigma_bias = nn.Parameter(copy.deepcopy(self.bias.data) / 2)
            self.phi_bias = nn.Parameter(copy.deepcopy(self.bias.data) / 2)
            self.bias = None
            self.bias_ = self.sigma_bias + self.phi_bias
        else:
            self.register_parameter('bias_', None)

    def forward(self, input):
        if self.bias is not None:
            return self._conv_forward(input, self.sigma_weight + self.phi_weight, self.sigma_bias + self.phi_bias)
        else:
            return self._conv_forward(input, self.sigma_weight + self.phi_weight, self.bias)

def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss

def init_param(m):
    if isinstance(m, nn.Conv2d) and isinstance(m, DecConv2d):
        nn.init.kaiming_normal_(m.sigma_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(m.phi_weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m




class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate):
        super().__init__()
        num_down = int(min(math.log2(data_shape[1]), math.log2(data_shape[2]))) - 3
        hidden_size = [16]
        for i in range(num_down + 1):
            hidden_size.append(16 * (2 ** i) * widen_factor)
        n = ((depth - 1) / (num_down + 1) - 1) / 2
        block = BasicBlock
        blocks = []
        blocks.append(nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False))
        blocks.append(NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate))
        for i in range(num_down):
            blocks.append(NetworkBlock(n, hidden_size[i + 1], hidden_size[i + 2], block, 2, drop_rate))
        blocks.append(nn.BatchNorm2d(hidden_size[-1]))
        blocks.append(nn.ReLU(inplace=True))
        #blocks.append(nn.AdaptiveAvgPool2d(1))
        #blocks.append(nn.Flatten())
        self.blocks = nn.Sequential(*blocks)
        #self.classifier = nn.Linear(hidden_size[-1], num_classes)

    def forward(self, x):
        x = self.blocks(x)
        #x = self.classifier(x)
        return x

    #def forward(self, input):
        #output = {}
        ##output['target'] = self.f(input['data'])
        #if 'loss_mode' in input:
            #if 'sup' in input['loss_mode']:
                #output['loss'] = loss_fn(output['target'], input['target'])
            #elif 'fix' in input['loss_mode'] and 'mix' not in input['loss_mode']:
                #aug_output = self.f(input['aug'])
                #output['loss'] = loss_fn(aug_output, input['target'].detach())
            #elif 'fix' in input['loss_mode'] and 'mix' in input['loss_mode']:
                #aug_output = self.f(input['aug'])
                #output['loss'] = loss_fn(aug_output, input['target'].detach())
                #mix_output = self.f(input['mix_data'])
                #output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                        #1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
            #else:
                #raise ValueError('Not valid loss mode')
        #else:
            #if not torch.any(input['target'] == -1):
                #output['loss'] = loss_fn(output['target'], input['target'])
        #return output


#def wresnet28x2(pretrained=False, momentum=None, track=False):
    #data_shape = cfg['data_shape']
    #target_size = cfg['target_size']
    #depth = cfg['wresnet28x2']['depth']
    #widen_factor = cfg['wresnet28x2']['widen_factor']
    #drop_rate = cfg['wresnet28x2']['drop_rate']
    
    
    #data_shape = (3,32,32)
    #target_size = 10
    #depth = 28
    #widen_factor = 4
    #drop_rate = 0.3
    #model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    #model.apply(init_param)
    #model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    #return model
#def wide_resnet28x2_features(pretrained=False, **kwargs):
    """Constructs a Wide ResNet-28-2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model = wide_resnet28_2(**kwargs)
    
    #if pretrained:
        #weights = WideResNet28_2_Weights.DEFAULT
        #state_dict = weights.get_state_dict(progress=True)
        
        # Remove the final fully connected layer weights and bias
        #state_dict.pop('fc.weight')
        #state_dict.pop('fc.bias')
        
        # Load the modified state dictionary
        #model.load_state_dict(state_dict, strict=False)
    
    # Remove the final fully connected layer
   # model = nn.Sequential(*list(model.children())[:-1])
    
    #return model

def wresnet28x8(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x8']['depth']
    widen_factor = cfg['wresnet28x8']['widen_factor']
    drop_rate = cfg['wresnet28x8']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet37x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet37x2']['depth']
    widen_factor = cfg['wresnet37x2']['widen_factor']
    drop_rate = cfg['wresnet37x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model
