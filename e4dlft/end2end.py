
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
        # NeuroVFM token embeddings
        tokens = self.encoder.embed(batch)   # [N,768]

        # Same pooling used during training
        vec = tokens.mean(dim=0).float().unsqueeze(0)

        logits = self.classifier(vec)        # [1,1]

        return logits