import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):

  def __init__(self):
    super(Inception, self).__init__()
    self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
    self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x) # 1-1) 128 -> 32 (1-final)

    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) # 2-1) 128 -> 128
    branch1x1_2 = self.branch1x1_2(branch1x1_pool) # 2-2) 128 -> 32 (2-final)

    branch3x3_reduce = self.branch3x3_reduce(x) # 3-1) 128 -> 24
    branch3x3 = self.branch3x3(branch3x3_reduce) # 3-2) 24 -> 32 (3-final)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x) # 4-1) 128 -> 24
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2) # 4-2) 24 -> 32 
    branch3x3_3 = self.branch3x3_3(branch3x3_2) # 4-3) 32 -> 32 (4-final) 

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3] # 32, 32, 32, 32
    return torch.cat(outputs, 1)


class FaceBoxesV2(nn.Module):

  def __init__(self, phase, size, num_classes):
    super(FaceBoxesV2, self).__init__()
    self.phase = phase
    self.num_classes = num_classes
    self.size = size
    
    self.conv1 = BasicConv2d(3, 8, kernel_size=3, stride=2, padding=1) # Conv + BN
    self.conv2 = BasicConv2d(8, 16, kernel_size=3, stride=2, padding=1)
    self.conv3 = BasicConv2d(16, 32, kernel_size=3, stride=2, padding=1)
    self.conv4 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
    self.conv5 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)
    
    self.inception1 = Inception() # x -> 4-each conv(output :32) -> concat[32, 32, 32, 32] -> (128, 1)
    self.inception2 = Inception()
    self.inception3 = Inception()
    
    self.conv6_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
    self.conv6_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    
    self.conv7_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
    self.conv7_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
    
    self.loc, self.conf = self.multibox(self.num_classes) # loc_layers, conf_layers 
    
    if self.phase == 'test':
        self.softmax = nn.Softmax(dim=-1)

    if self.phase == 'train':
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

  def multibox(self, num_classes):
    loc_layers = []
    conf_layers = []
    loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
    return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)
    
  def forward(self, x):
  
    sources = list()
    loc = list()
    conf = list()

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.inception1(x)
    x = self.inception2(x)
    x = self.inception3(x)
    sources.append(x)
    x = self.conv6_1(x)
    x = self.conv6_2(x)
    sources.append(x)
    x = self.conv7_1(x)
    x = self.conv7_2(x)
    sources.append(x)

    for (x, l, c) in zip(sources, self.loc, self.conf):
        loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
    loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
    conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

    if self.phase == "test":
      output = (loc.view(loc.size(0), -1, 4),
                self.softmax(conf.view(-1, self.num_classes)))
    else:
      output = (loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes))
  
    return output
