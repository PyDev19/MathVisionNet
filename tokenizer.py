import re
import json
from collections import defaultdict
from typing import Union, Tuple
from tqdm import tqdm

class LaTeXTokenizer:
    def __init__(self):
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.vocab['<SPC>'] = 2

    def tokenize(self, expression: str) -> list:
        parts = re.split(r'(\s+)', expression)
        tokens = []
        
        for part in parts:
            if part.isspace():
                tokens.extend(['<SPC>'] * len(part))
            else:
                tokens.extend(re.findall(r'\\[a-zA-Z]+|{|}|[0-9]+|[a-zA-Z]+|[^a-zA-Z0-9\s]', part))
        
        return tokens

    def build_vocab(self, expressions: list) -> None:
        for expression in tqdm(expressions, desc='Building vocabulary', unit='expression'):
            tokens = self.tokenize(expression)
            for token in tqdm(tokens, desc='Tokenizing', unit='token', leave=False):
                _ = self.vocab[token]
        
        json.dump(dict(self.vocab), open('data/vocab.json', 'w'), indent=4)
    
    def load_vocab(self, filepath: str) -> None:
        self.vocab = json.load(open(filepath))

    def encode(self, expression: str) -> Tuple[list, int]:
        tokens = self.tokenize(expression)
        lenghts = len(tokens)
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens], lenghts

    def decode(self, token_ids: list) -> str:
        reverse_vocab = {id: token for token, id in self.vocab.items()}
        decoded = ''.join(reverse_vocab.get(id, '<UNK>') for id in token_ids)
        
        return decoded.replace('<SPC>', ' ')
    
    def __call__(self, expression: Union[str, list]) -> Union[Tuple[list, int], str]:
        if isinstance(expression, str):
            return self.encode(expression)
        elif isinstance(expression, list):
            return self.decode(expression)
        else:
            raise ValueError('Input must be a string or a list of integers')

if __name__ == "__main__":
    import pandas as pd
    
    tokenizer = LaTeXTokenizer()
    # expressions = pd.read_csv('data/annotations.csv')['truth']
    # tokenizer.build_vocab(expressions)
    tokenizer.load_vocab('data/vocab.json')
    
    expression = input('Enter an expression: ')
    encoded, _ = tokenizer(expression)
    print(f'Encoded: {encoded}')
    
    decoded = tokenizer(encoded)
    print(f'Decoded: {decoded}')
