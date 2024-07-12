from torch.utils.data import Dataset
import torch
import os
import numpy as np
import pandas as pd
from scipy.ndimage import shift, rotate
from torchvision.transforms import functional as TF
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import normal_intensity_CT


class HybridSlice(Dataset):
    def __init__(self, csv_path='./csv/demo_split.csv',
                 binary='12', fold='train', augmentation=False, normal=False):

        df = pd.read_csv(csv_path)  # class 0, 1, 2
        df = df[df['split'] == fold]

        if binary == '01':
            df['class'] = (df['class'] > 0).astype(int)
        elif binary == '12':
            df = df[df['class'] != 2.5] # remove unknonw invasiveness samples
            df['class'] = (df['class'] > 1).astype(int)

        ind = df['path'].apply(lambda x: os.path.exists(x+'_AXN.npy') and os.path.exists(x+'_AXA.npy') and os.path.exists(x+'_AXV.npy'))
        self.npy_list = df['path'][ind].to_list()
        self.class_label_list = df['class'][ind].to_list()

        self.augmentation = augmentation
        self.normal = normal

    def __getitem__(self, index):
        npy_name = self.npy_list[index]

        imgN = np.load(npy_name + '_AXN.npy').squeeze()[np.newaxis, ::]
        imgA = np.load(npy_name + '_AXA.npy').squeeze()[np.newaxis, ::]
        imgV = np.load(npy_name + '_AXV.npy').squeeze()[np.newaxis, ::]

        if self.normal:
            imgN = normal_intensity_CT(imgN)
            imgA = normal_intensity_CT(imgA)
            imgV = normal_intensity_CT(imgV)

        if self.augmentation:
            imgN = self.random_augmentation(imgN)
            imgA = self.random_augmentation(imgA)
            imgV = self.random_augmentation(imgV)

        N_tensor = torch.tensor(imgN.copy(), dtype=torch.float32)
        A_tensor = torch.tensor(imgA.copy(), dtype=torch.float32)
        V_tensor = torch.tensor(imgV.copy(), dtype=torch.float32)

        y = self.class_label_list[index]
        y_tensor = torch.tensor(y, dtype=torch.long)

        return N_tensor, A_tensor, V_tensor, y_tensor

    def __len__(self):
        return len(self.npy_list)

    def random_augmentation(self, img):
        # Random shift
        delta_n = np.random.randn() * 0.1
        delta_m = np.random.randn() * 0.1
        img = shift(img, [0, int(img.shape[1] * delta_m), int(img.shape[2] * delta_n)], order=0, mode='constant')

        # Random rotation
        angle = np.random.uniform(-10, 10)  # Rotation angle between -10 and 10 degrees
        img = rotate(img, angle, axes=(1, 2), reshape=False, mode='constant', cval=0)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2)

        # Random vertical flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1)

        return img

    def get_classes_for_all_imgs(self):
        return self.class_label_list