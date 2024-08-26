import os
import json
import torch
from glob import glob
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.v2 import (
    Resize,
    Pad,
    CenterCrop,
    RandomRotation,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Compose,
    ColorJitter
)
from torchvision.io import read_image
from pytorch_lightning import LightningDataModule

class MathEquationsDataset(Dataset):
    def __init__(self, directory: str):
        self.images = glob(os.path.join(f'{directory}/background_images/', '*.jpg'))
        self.labels = json.load(glob(os.path.join(f'{directory}/background_images/', '*.json'))[0])
        
        self.transform = Compose([
            Resize((512, 512*4)),
            Pad((16, 16, 16, 16), fill=0, padding_mode='constant'),
            CenterCrop((512, 512*4)),
            RandomRotation(5),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        image_tensor = read_image(image)
        image_tensor = image_tensor.float() / 255.0
        image_tensor = self.transform(image_tensor)
        
        image_label = {}
        for label in self.labels:
            if label['filename'] == image.split('/')[-1]:
                image_label = label
                break
        
        tokenized_equation = image_label['image_data']['visible_char_map']
        tokenized_equation = torch.tensor(tokenized_equation)
        
        return image_tensor, tokenized_equation

class MathEquationsDatamodule(LightningDataModule):
    def __init__(self, directory: str, batch_size: int = 32, num_workers=2, test=False):
        super().__init__()
        
        self.directory = directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test = test
    
    def collate_fn(self, batch):
        images, equations = zip(*batch)
        
        images = torch.stack(images)
        equations = pad_sequence(equations, batch_first=True, padding_value=0.0)
        
        return images, equations
    
    def setup(self, stage=None):
        dataset = MathEquationsDataset(self.directory)
        
        if self.test:
            train_size = int(0.7 * len(dataset))
            val_size = int(0.2 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])
        else:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        if self.test:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.num_workers
            )
        else:
            raise ValueError('Test dataloader is not available for this module')
