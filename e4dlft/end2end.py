
import torch
import torch.nn as nn

class FCPart(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()

        self.architecture = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.architecture(x)

class End2End(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        #print(batch['img'].shape)
        #print(batch['series_cu_seqlens'])
        #raise SystemExit
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

class Cap4(nn.Module):
    def __init__(self, e2e, meta):
        super().__init__()
        self.e2e = e2e
        self.meta = meta

    def forward(self, img):
        batch = dict(self.meta)
        batch['img'] = img
        #print(img.shape)
        #raise SystemExit
        logit = self.e2e(batch)   # [N,768]
        print(f"logit from Cap4: {logit}")
        return logit


"""
class End2End(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        # NeuroVFM token embeddings
        print('hejhopp_0')
        print(batch.keys())
        print(batch['img'].shape)
        tokens = self.encoder.embed(batch)   # [N,768]
        print('___ *vec* ___')
        #print(tokens)
        print(tokens.shape)

        vec = tokens.mean(dim=0).float().unsqueeze(0)
        print('___ *vec* ___')
        #print(vec)
        print(vec.shape)

        logit = self.classifier(vec)
        print('___ *logit* ___')
        print(logit)
        print(logit.shape)
        return logit
"""