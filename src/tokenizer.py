import json
import os
import re
from glob import glob
from tqdm import tqdm
from utils import parse_inkml


class LaTeXTokenizer:
    def __init__(self, data_dir: str):
        self.token_re = re.compile(
            r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)"
        )

        self.vocab = (
            self._load_vocab(f"{data_dir}/vocab.json")
            if os.path.exists(f"{data_dir}/vocab.json")
            else self._build_vocab(data_dir)
        )
        self.rev_vocab = {idx: token for token, idx in self.vocab.items()}

        self.sos_token_id = self.vocab["<SOS>"]
        self.eos_token_id = self.vocab["<EOS>"]
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab["<UNK>"]
        self.spc_token_id = self.vocab["<SPC>"]

    def _load_vocab(self, vocab_file: str) -> dict:
        vocab = {}

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        return vocab

    def _tokenize(self, latex: str) -> list:
        tokens = []
        while latex:
            if latex.startswith("\\"):
                match = self.token_re.match(latex)
                if match:
                    token = match.group(0)
                    tokens.append(token)
                    latex = latex[len(token) :]
                else:
                    tokens.append("<UNK>")
                    latex = latex[1:]
            else:
                char = latex[0]
                char = "<SPC>" if char.isspace() else char
                tokens.append(char)
                latex = latex[1:]

        return tokens

    def _build_vocab(self, data_dir: str) -> dict:
        vocab = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<SPC>": 4,
        }

        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            if char not in vocab:
                vocab[char] = len(vocab)

        inkml_files = glob(f"{data_dir}/train/*.inkml", recursive=True)
        inkml_files.extend(glob(f"{data_dir}/synthetic/*.inkml", recursive=True))

        for file in tqdm(inkml_files, desc="Building vocabulary"):
            _, latex = parse_inkml(file)
            tokens = self._tokenize(latex)

            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        with open(f"{data_dir}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

        return vocab

    def encode(self, sequence: str) -> list:
        tokens = self._tokenize(sequence)
        tokens = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        tokens = [self.sos_token_id] + tokens + [self.eos_token_id]

        return tokens

    def decode(self, token_ids: list) -> str:
        for token in token_ids:
            if token not in self.rev_vocab:
                raise ValueError(f"Token ID {token} not found in vocabulary.")

        tokens = [self.rev_vocab.get(token, "<UNK>") for token in token_ids]
        decoded = "".join(
            " " if t == "<SPC>" else t
            for t in tokens
            if t not in ("<PAD>", "<SOS>", "<EOS>")
        )

        return decoded


if __name__ == "__main__":
    data_dir = "data/mathwriting-2024/"
    tokenizer = LaTeXTokenizer(data_dir)

    files = glob(f"{data_dir}/test/*.inkml", recursive=True)
    files.extend(glob(f"{data_dir}/valid/*.inkml", recursive=True))

    for file in tqdm(files, desc="Testing"):
        strokes, latex = parse_inkml(file)

        encoded = tokenizer.encode(latex)
        decoded = tokenizer.decode(encoded)

        assert (
            decoded == latex
        ), f"Decoded LaTeX does not match original for file {file}. Original: {latex}, Decoded: {decoded}"

    print("All test files passed the encoding and decoding test.")
