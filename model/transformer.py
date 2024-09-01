import torch
from torch import Tensor
from torch.nn import TransformerDecoder, TransformerDecoderLayer, Linear
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from model.embedding import ImageEmbeddings, TargetEmbeddings
from model.encoder import EncoderLayer
from scheduler import TransformerScheduler

class Image2LaTeXVisionTransformer(LightningModule):
    def __init__(self, hyperparameters: dict[str, int], optimizer_hyperparameters: dict):
        super().__init__()
        
        self.model_hyperparameters = hyperparameters
        self.optimizer_hyperparameters = optimizer_hyperparameters
        self.lr = optimizer_hyperparameters["learning_rate"]
        
        self.image_embeddings = ImageEmbeddings(hyperparameters)
        self.target_embeddings = TargetEmbeddings(hyperparameters)
        
        self.encoder = EncoderLayer(hyperparameters)
        
        self.decoder = TransformerDecoder(
            decoder_layer=TransformerDecoderLayer(
                d_model=hyperparameters["embedding_dim"],
                nhead=hyperparameters["num_heads"],
                dropout=hyperparameters["dropout"],
                dim_feedforward=hyperparameters["dim_feedforward"],
            ),
            num_layers=hyperparameters["num_decoder_layers"],
        )
        
        self.output_projection = Linear(hyperparameters["embedding_dim"], hyperparameters["vocab_size"])
        
        self.example_input_array = (torch.randn(1, 3, 128, 512), torch.randint(1, 91, (1, 100)))
        self.save_hyperparameters()
    
    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        
        return mask

    def create_padding_mask(self, sequences: Tensor) -> Tensor:
        return (sequences == 0)
    
    def forward(self, input_image: Tensor, target_sequence: Tensor) -> Tensor:
        image_embeddings = self.image_embeddings(input_image)
        target_embeddings = self.target_embeddings(target_sequence)
        
        encoder_output = self.encoder(image_embeddings)
        
        target_embeddings = target_embeddings.transpose(0, 1) # (sequence_length, batch_size, embedding_dim)
        decoder_image = encoder_output.transpose(0, 1) # (sequence_length, batch_size, embedding_dim)
        
        target_mask = self.generate_square_subsequent_mask(target_embeddings.size(0))
        target_padding_mask = self.create_padding_mask(target_sequence)
                
        decoder_output = self.decoder(
            target_embeddings,
            decoder_image,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask,
        )
        
        output_logits = self.output_projection(decoder_output)
        
        output_logits = output_logits.transpose(0, 1) # (batch_size, sequence_length, vocab_size)

        return output_logits

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input_image, target_sequence = batch
        output_logits = self(input_image, target_sequence[:, :-1])
        
        pad_mask = (target_sequence[:, 1:] != self.model_hyperparameters['padding_idx']).float()
        
        loss = cross_entropy(
            output_logits.view(-1, output_logits.size(-1)),
            target_sequence[:, 1:].contiguous().view(-1),
            reduction="none",
            label_smoothing=0.1
        )
        loss = (loss * pad_mask.view(-1)).sum() / pad_mask.sum()
        
        preds = torch.argmax(output_logits, dim=-1)
        correct = (preds == target_sequence[:, 1:]) * pad_mask.bool()
        acc = correct.sum() / pad_mask.sum()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input_image, target_sequence = batch
        output_logits = self(input_image, target_sequence[:, :-1])
        
        pad_mask = (target_sequence[:, 1:] != self.model_hyperparameters['padding_idx']).float()
        
        loss = cross_entropy(
            output_logits.view(-1, output_logits.size(-1)),
            target_sequence[:, 1:].contiguous().view(-1),
            reduction="none",
            label_smoothing=0.1
        )
        loss = (loss * pad_mask.view(-1)).sum() / pad_mask.sum()
        
        preds = torch.argmax(output_logits, dim=-1)
        correct = (preds == target_sequence[:, 1:]) * pad_mask.bool()
        acc = correct.sum() / pad_mask.sum()
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input_image, no_eos, no_sos = batch
        output_logits = self(input_image, no_eos)
        
        pad_mask = (no_sos != 0).float()
        
        loss = cross_entropy(
            output_logits.view(-1, output_logits.size(-1)),
            no_sos.contiguous().view(-1),
            reduction="none",
            label_smoothing=0.1
        )
        loss = (loss * pad_mask.view(-1)).sum() / pad_mask.sum()
        
        preds = torch.argmax(output_logits, dim=-1)
        correct = (preds == no_sos) * pad_mask.bool()
        acc = correct.sum() / pad_mask.sum()
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.lr,
            betas=self.optimizer_hyperparameters["betas"],
            eps=self.optimizer_hyperparameters["eps"]
        )

        scheduler = {
            'scheduler': TransformerScheduler(
                optimizer=optimizer,
                dim_embed=self.model_hyperparameters["embedding_dim"],
                warmup_steps=self.optimizer_hyperparameters["warmup_steps"]
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]
