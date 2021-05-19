from torch import nn
import torch
import math
import utils
from torchvision import transforms
import os
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
class EDSR(nn.Module):
    def __init__(self, conv=default_conv, synchronize_norm = True, device=None, scale_factor = 4):
        # synchronize_norm : True -- suppose that input was normalized mean 0.5, std 0.5   False -- suppose that input is in [0,1]
        super(EDSR, self).__init__()

        self.synchronize_norm = synchronize_norm
        self.device = device
        # should be changed
        n_resblocks = 32
        n_feats = 256
        kernel_size = 3 
        scale = scale_factor
        act = nn.ReLU(True)
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        if self.synchronize_norm:
            # When input is normalized mean 0.5, std 0.5
            x = x*0.5
            x += torch.tensor([[[0.0512]], [[0.0629]], [[0.096]]], device=self.device)
            
        else:
            # When input is in [0,1]
            x -= torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]], device=self.device)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x += torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]], device=self.device) # denormalization
        return x 
    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):

        res = self.body(x).mul(self.res_scale)
        res += x
        

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
def test():
    device = torch.device('cuda:1')
    gpu = 1
    print("GPU : ", gpu)
    model = EDSR(synchronize_norm=False, gpu=gpu, scale_factor=2).cuda(gpu)


    
    ckpt = torch.load(os.path.join('/mnt/nas/workspace/sr1', 'EDSR_x2.pt'))
    print("CKPT Loaded")
    model.load_state_dict(ckpt, strict=True)
    t = transforms.Compose([
            transforms.ToTensor()
        ])
    x = utils.pil_loader('/mnt/nas/data/track1/Corrupted-va-x/0802.png')
    x = t(x)
    x = torch.unsqueeze(x,0)
    x = x.cuda(gpu)
    print(x.size())
    x = model(x)[0]
    # x = transforms.Normalize(mean = mean_neg, std = std)(x)
    print(x.size())
    print(x[0][0][100])

    # y = utils.pil_loader('/mnt/nas/data/track1/DIV2K_valid_HR/0802.png')
    # y = transforms.ToTensor()(y).cuda(gpu)

    # psnr = -10*torch.log10(torch.mean((y-x)**2))
    # print(f"PSNR : {psnr}")
    utils.tensor_imsave(x, '/mnt/nas/workspace/sr1', 'test.png', denormalization=False)
if __name__ == '__main__':
    test()
