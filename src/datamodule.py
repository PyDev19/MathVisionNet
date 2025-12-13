import torch
from dataset import LaTeXDataset
from tokenizer import LaTeXTokenizer
from torchvision.transforms.v2 import (
    Resize,
    Compose,
    RandomRotation,
    GaussianBlur,
    Normalize,
    RandomApply,
)
from torch.utils.data import ConcatDataset, DataLoader


class LaTeXDataModule:
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        image_size: tuple = (128, 512),
        synthetic_dir: str = None,
        symbols_file: str = None,
        vocab_file: str = None,
    ):
        assert (
            symbols_file is not None and vocab_file is not None
        ), "symbols_file or vocab_file must be provided"

        self.tokenizer = LaTeXTokenizer(symbols_file, vocab_file=vocab_file)

        self.train_transform = Compose(
            [
                Resize(image_size),
                RandomRotation(degrees=3, fill=1.0),
                RandomApply([GaussianBlur(kernel_size=5, sigma=(0.3, 1.0))], p=0.3),
                Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.valid_transform = Compose(
            [
                Resize(image_size),
                Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_dataset = LaTeXDataset(
            train_dir, tokenizer=self.tokenizer, transform=self.train_transform
        )
        self.valid_dataset = LaTeXDataset(
            valid_dir, tokenizer=self.tokenizer, transform=self.valid_transform
        )
        self.test_dataset = LaTeXDataset(
            test_dir, tokenizer=self.tokenizer, transform=self.valid_transform
        )

        if synthetic_dir:
            self.synthetic_dataset = LaTeXDataset(
                synthetic_dir, tokenizer=self.tokenizer, transform=self.train_transform
            )
            self.train_dataset = ConcatDataset(
                [self.train_dataset, self.synthetic_dataset]
            )

    def get_train_loader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
        )

    def get_valid_loader(self, batch_size: int, test: bool = False) -> DataLoader:
        return DataLoader(
            self.valid_dataset if not test else self.test_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
        )
    
    def get_vocab_size(self) -> int:
        return len(self.tokenizer.vocab)

    def _collate_fn(self, batch: list) -> dict:
        images, labels = zip(*batch)

        images = torch.stack(images)

        max_len = max(label.size(0) for label in labels)
        padded_labels = torch.full(
            (len(labels), max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )

        for i, label in enumerate(labels):
            padded_labels[i, : label.size(0)] = label
            label_lengths.append(label.size(0))

        labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long)

        return images, labels_tensor, label_lengths
