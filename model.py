import torch
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Embedding, LSTM, Sequential, CrossEntropyLoss, Dropout
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EquationRecognitionModel(LightningModule):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(EquationRecognitionModel, self).__init__()
        self.save_hyperparameters()
        self.lr = 1e-3
        
        self.cnn = Sequential(
            Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.3),
            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.3),
            Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.3),
        )
        
        self.cnn_to_rnn = Linear(128 * 12 * 37, embed_size)
        
        self.embedding = Embedding(vocab_size, embed_size)
        self.lstm = LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.output_layer = Linear(hidden_size, vocab_size)
        
        self.loss = CrossEntropyLoss(ignore_index=0)
        self.accuracy = Accuracy(task='multiclass', num_classes=vocab_size, ignore_index=0)
        self.example_input_array = (torch.randn(1, 3, 100, 300), torch.randint(1, 45, (1, 10)), torch.tensor([10]))
        
    def forward(self, images, equations, lengths):
        cnn_features = self.cnn(images)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        cnn_features = self.cnn_to_rnn(cnn_features)
        cnn_features = cnn_features.unsqueeze(1).repeat(1, equations.size(1), 1)
        
        embedded_equations = self.embedding(equations)
        
        inputs = cnn_features + embedded_equations
        
        packed_inputs = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed_inputs)
        
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        outputs = self.output_layer(outputs)
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        images, equations, lengths = batch
        
        outputs = self(images, equations[:, :-1], lengths - 1)
        targets = equations[:, 1:]
        
        loss = self.loss(outputs.transpose(2, 1), targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        preds = torch.argmax(outputs, dim=2)
        acc = self.accuracy(preds, targets)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, equations, lengths = batch
        
        outputs = self(images, equations[:, :-1], lengths - 1)
        targets = equations[:, 1:]
        
        loss = self.loss(outputs.transpose(2, 1), targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        preds = torch.argmax(outputs, dim=2)
        acc = self.accuracy(preds, targets)
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
