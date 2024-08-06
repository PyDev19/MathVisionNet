import pandas as pd
from dataset import EquationsImageDataset, collate_fn
from torch.utils.data import DataLoader, random_split

data = pd.read_csv('data/annotations.csv')
dataset = EquationsImageDataset(data)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=16, pin_memory=True)