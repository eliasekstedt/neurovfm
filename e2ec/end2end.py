
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

class End2EndForCaptum(nn.Module):
    def __init__(self, end2end, batch_template):
        super().__init__()
        self.end2end = end2end
        self.batch_template = batch_template

    def forward(self, x):
        batch = dict(self.batch_template)
        batch["img"] = x


        ###
        n = self.batch_template["img"].shape[0]
        batch["coords"] = torch.cat(
            [batch["coords"], batch["coords"]],
            )
        batch["series_cu_seqlens"] = torch.tensor(
            [0, n, 2*n],
            dtype=torch.int32,
            device=batch["coords"].device,
        )
        batch["series_max_len"] = n
        batch["study_cu_seqlens"] = torch.tensor(
            [0, n, 2*n],
            dtype=torch.int32,
            device=batch["coords"].device,
        )
        batch["study_max_len"] = n
        ###
        print("img:", batch["img"].shape)
        print("coords:", batch["coords"].shape)
        print("series_cu_seqlens:", batch["series_cu_seqlens"])
        print("series_max_len:", batch["series_max_len"])
        print("study_cu_seqlens:", batch["study_cu_seqlens"])
        print("study_max_len:", batch["study_max_len"])
        print(batch["series_cu_seqlens"])
        print(batch["study_cu_seqlens"])
        """
        raise SystemExit
        """
        out = self.end2end(batch)

        print("FINAL RETURN SHAPE:", out.shape)

        #raise SystemExit
        return out