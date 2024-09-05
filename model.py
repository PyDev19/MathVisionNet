import torch
import json
from typing import Tuple
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import AdamW
from image2latex.scheduler import TransformerLRScheduler
from image2latex.transformer import VisionTransformerDecoder
from pytorch_lightning import LightningModule
from torchmetrics.functional.text import char_error_rate, bleu_score

class Image2LatexModel(LightningModule):
    def __init__(self, model_hyperparameters: dict[str, int], optimizer_hyperparameters: dict, vocab_file: str):
        super().__init__()
        self.model = VisionTransformerDecoder(model_hyperparameters)
        self.vocab = json.load(open(vocab_file, "r"))
        
        self.lr = optimizer_hyperparameters["learning_rate"]
        self.optimizer_hyperparameters = optimizer_hyperparameters
        
        self.example_input_array = (torch.randn(1, 3, 128, 512), torch.randint(1, 91, (1, 100)))
        self.save_hyperparameters()
    
    def forward(self, input_image: Tensor, target_sequence: Tensor) -> Tensor:
        return self.model(input_image, target_sequence)

    def decode_sequence(self, sequence: Tensor) -> str:
        decoded_sequence = "".join([self.vocab[str(token)] for token in sequence if token != 0])
        decoded_sequence = decoded_sequence.replace("[SOS]", "")
        
        return decoded_sequence
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input_image, no_eos, no_sos = batch
        output_logits = self(input_image, no_eos)
        
        pad_mask = (no_sos != 0).float()
        
        loss = cross_entropy(
            output_logits.contiguous().view(-1, output_logits.size(-1)),
            no_sos.contiguous().view(-1),
            reduction="none",
            label_smoothing=0.1
        )
        loss = (loss * pad_mask.view(-1)).sum() / pad_mask.sum()
        
        preds = torch.argmax(output_logits, dim=-1)
        correct = (preds == no_sos) * pad_mask.bool()
        acc = correct.sum() / pad_mask.sum()
        
        decoded_preds = [self.decode_sequence(pred) for pred in preds]
        decoded_targets = [self.decode_sequence(target) for target in no_sos]
        
        char_error = char_error_rate(decoded_preds, decoded_targets)
        bleu = bleu_score(decoded_preds, decoded_targets)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_char_error_rate", char_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_bleu_score", bleu, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input_image, no_eos, no_sos = batch
        output_logits = self(input_image, no_eos)
        
        pad_mask = (no_sos != 0).float()
        
        loss = cross_entropy(
            output_logits.contiguous().view(-1, output_logits.size(-1)),
            no_sos.contiguous().view(-1),
            reduction="none",
            label_smoothing=0.1
        )
        loss = (loss * pad_mask.view(-1)).sum() / pad_mask.sum()
        
        preds = torch.argmax(output_logits, dim=-1)
        correct = (preds == no_sos) * pad_mask.bool()
        acc = correct.sum() / pad_mask.sum()
        
        decoded_preds = [self.decode_sequence(pred) for pred in preds]
        decoded_targets = [self.decode_sequence(target) for target in no_sos]
        
        char_error = char_error_rate(decoded_preds, decoded_targets)
        bleu = bleu_score(decoded_preds, decoded_targets)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_char_error_rate", char_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_bleu_score", bleu, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        input_image, no_eos, no_sos = batch
        output_logits = self(input_image, no_eos)
        
        pad_mask = (no_sos != 0).float()
        
        loss = cross_entropy(
            output_logits.contiguous().view(-1, output_logits.size(-1)),
            no_sos.contiguous().view(-1),
            reduction="none",
            label_smoothing=0.1
        )
        loss = (loss * pad_mask.view(-1)).sum() / pad_mask.sum()
        
        preds = torch.argmax(output_logits, dim=-1)
        correct = (preds == no_sos) * pad_mask.bool()
        acc = correct.sum() / pad_mask.sum()
        
        decoded_preds = [self.decode_sequence(pred) for pred in preds]
        decoded_targets = [self.decode_sequence(target) for target in no_sos]
        
        char_error = char_error_rate(decoded_preds, decoded_targets)
        bleu = bleu_score(decoded_preds, decoded_targets)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_char_error_rate", char_error, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_bleu_score", bleu, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            betas=self.optimizer_hyperparameters["betas"],
            eps=self.optimizer_hyperparameters["eps"],
            weight_decay=self.optimizer_hyperparameters["weight_decay"]
        )

        scheduler = {
            'scheduler': TransformerLRScheduler(
                optimizer=optimizer,
                dim_embed=self.model_hyperparameters["embedding_dim"],
                warmup_steps=self.optimizer_hyperparameters["warmup_steps"]
            ),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]