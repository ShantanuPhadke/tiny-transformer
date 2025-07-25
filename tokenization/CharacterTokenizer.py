import torch

class CharacterTokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.char_to_index = {char: idx for idx, char in enumerate(vocabulary.chars)}
        self.index_to_char = {idx: char for idx, char in enumerate(vocabulary.chars)}

    def encode(self, raw_text):
        return torch.tensor([self.char_to_index[char] for char in raw_text], dtype=torch.long)

    def decode(self, encoding):
        return ''.join([self.index_to_char[num.item()] for num in encoding])