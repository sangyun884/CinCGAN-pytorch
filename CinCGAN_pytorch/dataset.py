import torch.utils.data as data
import random
from PIL import Image
from torchvision import transforms
import os
import os.path
from utils import *
from transforms import *

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(dir, extensions, label):
    # Returns the tuples (path, label) of data
    # Now, label is same with fname
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                item = (path, fname)
                images.append(item)

    return images

class DatasetFolder(data.Dataset):
    def __init__(self, root, label, extensions = ['.jpg', '.png', '.jpeg', '.bmp'], transform=None, return_two_img = False, big_imsize = 128, scale_factor = 4):
        # root -- Dataset folder path
        # label -- 0 / 1
        # extensions -- list of alowed extensions
        # transform -- Transform applied to image
        # return_two_img -- Whether to return both big and small imgs or not
        # big_imsize -- Size of bigger img. Matters only if return_two_img is True
        # scale_factor -- SR scale factor. Matters only if return_two_img is True
        samples = make_dataset(root, extensions, label)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        
        self.root = root
        self.extensions = extensions
        self.samples = samples
        self.transform = transform
        self.return_two_img = return_two_img
        self.big_imsize = big_imsize
        self.scale_factor = scale_factor
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = pil_loader(path)
        


        if self.return_two_img:
            t1 = transforms.Compose([
            Random90Rot(),
            transforms.RandomCrop((self.big_imsize,self.big_imsize))])

            t2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

            sample = t1(sample)
            sample_big = t2(sample)
            sample_small = transforms.Resize((self.big_imsize//self.scale_factor, self.big_imsize//self.scale_factor), Image.BICUBIC)(sample)
            sample_small = t2(sample_small)

            return sample_big, sample_small, target

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
    
    def __len__(self):
        return len(self.samples)

