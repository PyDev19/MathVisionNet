import json


class LaTeXTokenizer:
    def __init__(self, symbols_file: str, vocab_file: str = None):
        self.vocab = (
            self._load_vocab(vocab_file)
            if vocab_file
            else self._build_vocab(symbols_file)
        )
        self.rev_vocab = {idx: token for token, idx in self.vocab.items()}

        self.sos_token_id = self.vocab["<SOS>"]
        self.eos_token_id = self.vocab["<EOS>"]
        self.pad_token_id = self.vocab["<PAD>"]
        self.unk_token_id = self.vocab["<UNK>"]

    def _load_vocab(self, vocab_file: str) -> dict:
        vocab = {}

        with open(vocab_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.strip()
                vocab[token] = idx

        return vocab

    def _build_vocab(self, symbols_file: str) -> dict:
        vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "<SPC>": 4}

        for char_code in range(32, 127):
            char = chr(char_code)
            if char not in vocab and not char.isspace():
                vocab[char] = len(vocab)

        with open(symbols_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines:
                line_json = json.loads(line)
                label = line_json["label"]

                if label not in vocab:
                    vocab[label] = len(vocab)

            with open("vocab.json", "w") as f:
                json.dump(vocab, f, indent=4)

            f.close()

        return vocab

    def encode(self, sequence: str) -> list:
        temp_token = ""
        tokens = []

        for char in sequence:
            if char.isspace():
                tokens.append(self.vocab["<SPC>"])

            temp_token += char

            if temp_token in self.vocab:
                tokens.append(self.vocab[temp_token])
                temp_token = ""

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
    tokenizer = LaTeXTokenizer("mathwriting-2024-excerpt/symbols.jsonl")

    example_string = r"\overline{hu^{2}}+\frac{1}{2}k_{ap}g_{z}h^{2}"

    encoded = tokenizer.encode(example_string)
    decoded = tokenizer.decode(encoded)

    print(example_string)
    print(encoded)
    print(decoded)
