import os
import ast
import pandas as pd
import torch
from glob import glob
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.v2 import (
    Resize,
    Compose,
    ColorJitter,
    RandomAffine,
    GaussianBlur,
    RandomInvert
)
from torchvision.io import read_image
from pytorch_lightning import LightningDataModule

class MathEquationsDataset(Dataset):
    def __init__(self, directory: str, transforms: Compose, image_size=(128, 512)):
        self.images = glob(os.path.join(f'{directory}/images/', '*.jpg'))
        self.labels = pd.read_csv(f'{directory}/annotations.csv')
        
        self.transform = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        basename = os.path.basename(image)
        
        image_tensor = read_image(image)
        image_tensor = image_tensor.float() / 255.0
        image_tensor = self.transform(image_tensor)
        
        image_data = self.labels.loc[self.labels['filenames'] == basename]['image_data']
        image_data = ast.literal_eval(image_data.values[0])
        
        tokenized_equation = image_data['visible_char_map']
        tokenized_equation = [92] + tokenized_equation + [93]
        
        tokenized_equation = torch.tensor(tokenized_equation)
        
        return image_tensor, tokenized_equation

class MathEquationsDatamodule(LightningDataModule):
    def __init__(self, directory: str, image_size, batch_size: int = 32, num_workers=2):
        super().__init__()
        
        self.directory = directory
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        self.train_transform = Compose([
            Resize(image_size),
            RandomInvert(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5, hue=0.5),
            RandomAffine(degrees=2, translate=(0.02, 0.02)),
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        ])
        
        self.val_dataloader = Compose([Resize(image_size)])
    
    def collate_fn(self, batch):
        images, equations = zip(*batch)
        
        images = torch.stack(images)
        equations_without_eos = [equation[:-1] for equation in equations]
        equations_without_sos = [equation[1:] for equation in equations]
        
        no_eos = pad_sequence(equations_without_eos, batch_first=True, padding_value=0)
        no_sos = pad_sequence(equations_without_sos, batch_first=True, padding_value=0)
        
        return images, no_eos, no_sos
    
    def setup(self, stage=None):
        dataset = MathEquationsDataset(self.directory, self.image_size)
        
        train_size = int(0.6 * len(dataset))
        val_size = int(0.2 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])
        
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform
        self.test_dataset.dataset.transform = self.val_transform

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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.num_workers
        )
