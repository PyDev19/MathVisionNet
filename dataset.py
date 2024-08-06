import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose, ToDtype, CenterCrop, ToTensor, RandomRotation, ColorJitter, Normalize
from PIL import Image
from tokenizer import LaTeXTokenizer

class EquationsImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        self.tokenizer = LaTeXTokenizer()
        self.tokenizer.load_vocab('data/vocab.json')
        
        self.image_transform = Compose([
            CenterCrop((100, 300)),
            RandomRotation(5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ToDtype(torch.float32, scale=True),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = f'data/train/{self.data['path'][idx]}'
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        
        equation = self.data['truth'][idx]
        encoded_equation, length = self.tokenizer(equation)
        equation_tensor = torch.tensor(encoded_equation, dtype=torch.long)
        length_tensor = torch.tensor(length, dtype=torch.long)
        
        return image_tensor, equation_tensor, length_tensor


def collate_fn(batch):
    images, equations, lengths = zip(*batch)
    
    images = torch.stack(images)
    padded_equations = pad_sequence(equations, batch_first=True, padding_value=0)
    lengths = torch.stack(lengths)
    
    return images, padded_equations, lengths

