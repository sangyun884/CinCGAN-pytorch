
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from utils import *


class ResnetGenerator(nn.Module):
    #Generator architecture
    def __init__(self, input_nc=3, output_nc=3, inter_nc=64, n_blocks=6, img_size = 32, use_bias = False, rs_norm = 'BN', padding_type = 'zero', dsple = False, scale_factor=4):
        # input_nc(int) -- The number of channels of input img
        # output_nc(int) -- The number of channels of output img
        # inter_nc(int) -- The number of filters of intermediate layers
        # n_blocks(int) -- The number of resnet blocks
        # img_size(int) -- Input image size
        # use_bias(bool) -- Whether to use bias on conv layer or not
        # rs_norm(str) -- The type of normalization method of ResnetBlock. BN : Batch Normalization, IN : Instance Normalization, else : none
        # padding_type(str) -- The name of padding layer: reflect | replicate | zero
        # dsple(bool) -- Whether to downsample or maintain input image. Set it true for G3.
        # scale_factor(int) -- Scale factor, 2 / 4
        super(ResnetGenerator, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inter_nc = inter_nc
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.use_bias = use_bias
        self.rs_norm = rs_norm
        self.padding_type = padding_type
        self.dsple = dsple
        self.scale_factor = scale_factor
        
        # Input blocks
        InBlock = []
        
        InBlock += [nn.Conv2d(input_nc, inter_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        InBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple and self.scale_factor==4 else 1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)] #changed
        InBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2 if self.dsple else 1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        
        # ResnetBlocks
        ResnetBlocks = []
        
        for i in range(n_blocks):
            ResnetBlocks += [ResnetBlock(inter_nc, self.padding_type, self.rs_norm, self.use_bias)]
        
        # Output block
        OutBlock = []
        
        OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        OutBlock += [nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=1, padding=1, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        OutBlock += [nn.Conv2d(inter_nc, output_nc, kernel_size=7, stride=1, padding=3, bias=self.use_bias),
                     nn.LeakyReLU(0.2)]
        
        self.InBlock = nn.Sequential(*InBlock)
        self.ResnetBlocks = nn.Sequential(*ResnetBlocks)
        self.OutBlock = nn.Sequential(*OutBlock)
    def forward(self,x):
        out = self.InBlock(x)
        out = self.ResnetBlocks(out)
        out = self.OutBlock(out)
        
        return out
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_type, use_bias):
        # dim(int) -- The number of channels in the resnet blocks
        # padding_type(str) -- The name of padding layer: reflect | replicate | zero
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias -- Whether to use bias on conv layer or not
        super(ResnetBlock, self).__init__()
        
        conv_block = []
        
        # Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            
        if norm_type=='BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type=='IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.LeakyReLU(0.2)]

        
        # Padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                       norm_layer(dim), 
                       nn.LeakyReLU(0.2)]
        
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = self.conv_block(x)
        
        # Skip connection
        out = out + x
        
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, norm_type = 'BN', use_bias = True, is_inner=True, scale_factor=4):
        # input_nc(int) -- The number of channels of input img
        # norm_type(str) -- The type of normalization method. BN : Batch Normalization, IN : Instance Normalization, else : none
        # use_bias(bool) -- Whether to use bias or not
        # is_inner(bool) -- True : For inner cycle, False : For outer cycle
        # scale_factor(int) -- Scale factor, 2 / 4

        super(Discriminator, self).__init__()
        
        if norm_type=='BN':
            norm_layer = nn.BatchNorm2d
            use_bias = False # There is no need to use bias because BN already has shift parameter.
        elif norm_type=='IN':
            norm_layer = nn.InstanceNorm2d
        else:
            raise NotImplementedError('Normalization [%s] is not implemented' % norm_type)
        
        if is_inner == True:
            s = 1
        elif is_inner == False:
            s = 2
        else:
            raise NotImplementedError('is_inner must be boolean.')
        
        nfil_mul = 64
        p=0 # Why 1???
        layers = []
        layers += [nn.Conv2d(input_nc, nfil_mul, kernel_size=4, stride = 2 if is_inner==True and scale_factor==2 else s, padding=p, bias=use_bias), 
                       nn.LeakyReLU(0.2)] # changed
        layers += [nn.Conv2d(nfil_mul, nfil_mul*2, kernel_size=4, stride = s, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*2), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*2, nfil_mul*4, kernel_size=4, stride = s, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*4), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*4, nfil_mul*8, kernel_size=4, stride = 1, padding=p, bias=use_bias), 
                       norm_layer(nfil_mul*8), 
                       nn.LeakyReLU(0.2)]
        layers += [nn.Conv2d(nfil_mul*8, 1, kernel_size=4, stride = 1, padding=p, bias=use_bias), 
                       nn.LeakyReLU(0.2)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.layers(x)
        
        return out # Predicted values of each patches

        

def test():
    from torchvision import transforms
    import matplotlib.pyplot as plt

    #device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load image
    img = pil_loader("/mnt/nas/data/track1/Corrupted-te-x/0917.png")
    
    # Transform image to torch Tensor. Normalized to 0~1 automatically.
    img = transforms.Resize((128,128))(img)
    img = transforms.ToTensor()(img).to(device)
    
    print("Input shape : ",img.shape)

    # Feed to generator
    img = torch.unsqueeze(img,0)
    G1 = ResnetGenerator(dsple=True).to(device)
    fakeimgs = G1(img)

    print("Fake image shape : ", fakeimgs.shape)

    # Feed to discriminator
    D1 = Discriminator(is_inner=True).to(device)
    out = D1(fakeimgs)
    print(out.shape)



