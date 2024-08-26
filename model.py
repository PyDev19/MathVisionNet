from typing import Union
import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from torch.nn import (
    Module,
    Sequential,
    Embedding,
    TransformerDecoder,
    TransformerDecoderLayer,
    Linear,
    Conv2d,
    ReLU,
    SiLU,
    MaxPool2d,
    BatchNorm2d
)
from torch.nn.functional import cross_entropy
from torchmetrics.functional import accuracy

class PositionalEncoding(Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        positional_encoding = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        positional_encoding = positional_encoding.unsqueeze(1)
        
        self.register_buffer('positional_encoding', positional_encoding)
        
    def forward(self, x):
        x = x + self.positional_encoding[:x.size(0), :]
        
        return x
    
class Conv2dBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu'):
        super(Conv2dBlock, self).__init__()
        
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        if activation == 'relu':
            self.activation = ReLU()
        if activation == 'silu':
            self.activation = SiLU()
        
        self.batchnorm = BatchNorm2d(num_features=out_channels)
        self.maxpool = MaxPool2d(kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.batchnorm(x)
        x = self.maxpool(x)
        
        return x

class Image2LaTeXTransformer(LightningModule):
    def __init__(self, hyperparameters: dict[str, Union[int, list[int]]]):
        super(Image2LaTeXTransformer, self).__init__()
        channels = hyperparameters['cnn_channels']
        output_dim = channels[-1]
        
        self.embedding = Embedding(hyperparameters['vocab_size'], output_dim)
        self.positional_encoding = PositionalEncoding(output_dim)
        
        self.cnn = Sequential()

        for i in range(len(channels)-1):
            self.cnn.add_module(f'conv2d_block_{i}', Conv2dBlock(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=1,
                stride=1,
                padding=0,
                activation=hyperparameters['activation']
            ))
        
        self.cnn_to_decoder = Linear(512 * 128 * 32, output_dim)
        
        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=output_dim,
                nhead=hyperparameters['num_heads'],
                dim_feedforward=hyperparameters['dim_feedforward'],
                dropout=hyperparameters['dropout']
            ),
            num_layers=hyperparameters['num_decoder_layers']
        )
        
        self.output_layer = Linear(output_dim, hyperparameters['vocab_size'])
        
        self.save_hyperparameters()
        self.example_input_array = (torch.rand(1, 3, 512, 512*4), torch.randint(1, hyperparameters['vocab_size'], (1, hyperparameters['vocab_size'])))
    
    def forward(self, images: Tensor, equations: Tensor):
        image_features: Tensor = self.cnn(images)
        
        image_features = image_features.view(image_features.size(0), -1)
        image_features = self.cnn_to_decoder(image_features)
        
        image_features = image_features.view(-1, image_features.size(0), image_features.size(1))
        
        target_embedding = self.embedding(equations)
        target_embedding = self.positional_encoding(target_embedding)
        
        output = self.decoder(target_embedding, image_features)
        output = self.output_layer(output)
        
        return output
