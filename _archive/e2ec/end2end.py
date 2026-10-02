
import torch
import torch.nn as nn

class ANN(nn.Module):
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
        out = self.architecture(x)
        print(out); raise SystemExit
        return out

class End2End(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        # NeuroVFM token embeddings
        tokens = self.encoder.embed(batch)   # [N,768]

        baseline_tokens = tokens[:208]
        input_tokens = tokens[208:]

        baseline_vec = baseline_tokens.mean(dim=0).float().unsqueeze(0)
        input_vec = input_tokens.mean(dim=0).float().unsqueeze(0)

        baseline_logit = self.classifier(baseline_vec)
        input_logit = self.classifier(input_vec)

        return torch.cat([baseline_logit, input_logit], dim=1)
        """
        # Same pooling used during training
        vec = tokens.mean(dim=0).float().unsqueeze(0)

        logits = self.classifier(vec)        # [1,1]
        return logits
        """




