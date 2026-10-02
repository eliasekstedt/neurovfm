
import pandas as pd
import torch.nn as nn
import torch

import contextlib
import io


class Model(nn.Module):
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

class CombinedModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()

        self.encoder = encoder      # NeuroVFM EncoderPipeline
        self.classifier = classifier

    def forward(self, batch):
        # NeuroVFM token embeddings
        tokens = self.encoder.embed(batch)   # [N,768]

        # Same pooling used during training
        vec = tokens.mean(dim=0).float().unsqueeze(0)

        logits = self.classifier(vec)        # [1,1]

        return logits


device = 'cuda:0'

from neurovfm.pipelines.encoder import load_encoder
std_encoder, preproc = load_encoder("ckpt")

from e4dlft_config import *
from e4dlft_run import get_encoder
e4dlft_encoder = get_encoder(
    vit_params=cfg.vit_params,
    device='cuda:0',
    fpath_encoderStatedict=cfg.fpath_encoderStatedict,
)

fpath_ann = cfg.fpath_FCStatedict
fpath_evaldata = cfg.fpath_evalData
ann = Model(embed_dim=768, dropout=0.5)
ann.load_state_dict(torch.load(fpath_ann))
mod = cfg.modality

logits_lst = []
for encoder in [std_encoder, e4dlft_encoder]:
    model = CombinedModel(
        encoder=encoder,
        classifier=ann,
    ).to(device)
    model.eval()

    from config_LHg import *
    df = pd.read_csv(cfg.fpath_manifest)
    id = pd.read_csv(fpath_evaldata)['id'].to_list()[0]

    #with torch.inference_mode(): # <- change !!!!!!!!!!!!!!!!!
    fpath_nii = cfg.dpath_hgg / id / f"{id}_{mod}.nii.gz"
    with contextlib.redirect_stdout(io.StringIO()):
        batch = preproc.load_study(
            fpath_nii,
            modality="mri",
        )

        #raise SystemExit
        logits = model(batch)
        logits_lst.append(logits)

print(torch.allclose(logits_lst[0], logits_lst[1]))
for logits in logits_lst:
    print(logits.shape)
    print(logits[:5])
    


