from torchvision import transforms
import random
from PIL import Image
class Random90Rot(object):
    def __call__(self, img):
        angle_list = [0,-90,-180,90]
        angle = random.choice(angle_list)

        return transforms.functional.rotate(img, angle)
class ResizeTensor(object):
    def __init__(self, size, interpolation = Image.BICUBIC):
        # size -- desirable output size 
        # interpolation -- interpolation method
        # Values in tensor must be [0,1]
        
        self.size = size
        self.interpolation = interpolation
    def __call__(self,x):
        # Return resized tensor
        x = x*255
        x = transforms.ToPILImage(x)
        x = transforms.Resize(self.size, self.interpolation)(x)
        x = transforms.ToTensor(x)

        return x
    
class Crop(object):
    def __init__(self, max_hw):
        self.max_hw = max_hw
    def __call__(self,x):
        if x.size[0]>=self.max_hw and x.size[1]>=self.max_hw:
            x = transforms.CenterCrop((self.max_hw,self.max_hw))(x)
        return x