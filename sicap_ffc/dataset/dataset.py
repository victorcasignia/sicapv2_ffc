from torch.utils.data import Dataset
import pandas as pd
import math
from torchvision.io import read_image
import os

class SicapDataset(Dataset):
    def __init__(self, file_name, img_dir, transform=None, target_transform=None, preload=True):        
        df = pd.read_excel(file_name)
        df['Gleason_label'] = (df['NC'].map(str) + df['G3'].map(str) + df['G4'].map(str) + df['G5'].map(str))
        df['Gleason_label'] = df['Gleason_label'].apply(lambda x: math.log(int(x, 2),2))
        
        x = df['image_name']
        y = df['Gleason_label'].astype(int)
        
        print(y.unique())

        x.reset_index(drop = True)
        y.reset_index(drop = True)
        self.x=x
        self.y=y
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = img_dir
        
        self.preload = preload
        if self.preload:
            self.x_images = self.x.progress_apply(lambda img: self.transform(read_image(os.path.join(self.img_dir, img))))

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        if self.preload:
            image = self.x_images[idx]
        else: 
            img_path = os.path.join(self.img_dir, self.x[idx])
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
        label = self.y[idx]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label