
import torch

class FCPart(torch.nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()

        self.architecture = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.architecture(x)

class End2End(torch.nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        tokens = self.encoder.embed(batch)   # [N,768]
        tokens = self.reconstruct(tokens, batch)
        tokens = tokens.mean(dim=1).float()#.unsqueeze(0)
        logit = self.classifier(tokens)
        return logit

    def reconstruct(self, tokens, batch):
        cu = batch["series_cu_seqlens"]
        studies = [
            tokens[cu[i]:cu[i+1]]
            for i in range(len(cu)-1)
        ]
        [print(s.shape) for s in studies]
        studies = torch.stack(studies, dim=0)
        print(studies.shape)
        return studies