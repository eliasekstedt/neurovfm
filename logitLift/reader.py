
import torch


class Reader:
    def __init__(self, fpath):
        self.tokens = torch.load(fpath, weights_only=False)['embvec'].float()#[n, 768]

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        x = self.tokens[idx, :]
        return x

