
import torch


class Reader:
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        x = self.tokens[idx, :]
        return idx, x

