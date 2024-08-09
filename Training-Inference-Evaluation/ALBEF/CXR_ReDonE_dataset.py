import numpy as np
import torch
from PIL import Image
import re
# import h5py
from torch.utils import data
from torch.utils.data import Dataset
import os
#Adapted cxr-repair
#input: .h5 file containing the images
# class CXRTestDataset_h5(data.Dataset):
#     def __init__(self, img_path, transform=None):
#         super().__init__()
#         self.img_dset = h5py.File(img_path, 'r')['cxr']
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.img_dset)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         img = self.img_dset[idx]
#         img = np.expand_dims(img, axis=0)
#         img = np.repeat(img, 3, axis=0)
#         img = torch.from_numpy(img)
#         if self.transform:
#             img = self.transform(img)
        
#         return img

#Adapted cxr-repair
#input: files containing paths to the image files
class CXRTestDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super().__init__()
        self.files = sorted(
    [os.path.join(img_path, fname) for fname in os.listdir(img_path) if fname.lower().endswith(('png', 'jpg', 'jpeg'))],
    key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', os.path.basename(x))]
)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fpath = self.files[idx]
        desired_size = 256
        img = Image.open(fpath).convert('L')  # Convert image to grayscale ('L')
        old_size = img.size
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, Image.ANTIALIAS)
        new_img = Image.new('L', (desired_size, desired_size))
        new_img.paste(img, ((desired_size - new_size[0]) // 2,
                            (desired_size - new_size[1]) // 2))
        img = np.asarray(new_img, np.float64)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img)
        if self.transform:
            img = self.transform(img)
        return img