
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch
from tqdm import tqdm

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


#from ctests import nothing


device = 'cuda:0'

from neurovfm.pipelines.encoder import load_encoder
encoder, preproc = load_encoder("ckpt")

from e4dlft_config import *
from e4dlft_run import get_encoder
e4dlft_encoder = get_encoder(
    vit_params=cfg.vit_params,
    device='cuda:0',
    fpath_weights=cfg.fpath_weights,
)

fpath_ann = Path('../mmMRI/_v5_NeuroVFM/run/report_better/22_07_36_00/model.pth')
fpath_evaldata = Path('../mmMRI/_v5_NeuroVFM/run/report_better/22_07_36_00/eval.csv')
ann = Model(embed_dim=768, dropout=0.5)
ann.load_state_dict(torch.load(fpath_ann))

std_model = CombinedModel(
    encoder=encoder,
    classifier=ann,
).to(device)
std_model.eval()

mod = 't1ce'
from config_LHg import *
df = pd.read_csv(cfg.fpath_manifest)
ids_unseen = pd.read_csv(fpath_evaldata)['id'].to_list()

results = {'id':[], 'label':[], 'prob':[], 'accurate':[], 'logit':[]}
with torch.inference_mode():
    for label in range(2):
        ids = df[df['label']==label]['id'].to_list()
        ids = [id for id in ids if id in ids_unseen]
        dpath_x = [cfg.dpath_lgg, cfg.dpath_hgg][label]
        for id in tqdm(ids):
            fpath_nii = dpath_x / id / f"{id}_{mod}.nii.gz"
            with contextlib.redirect_stdout(io.StringIO()):
                batch = preproc.load_study(
                    fpath_nii,
                    modality="mri",
                )


            #raise SystemExit
            logits = std_model(batch)
            pred = (logits > 0).int().item()
            accurate = int(pred == label)
            
            results['id'].append(id)
            results['label'].append(label)
            results['prob'].append(pred)
            results['accurate'].append(accurate)
            results['logit'].append(logits.item())


results = pd.DataFrame(results)
results.to_csv('verify_combined_model.csv', index=False)
print(f'accuracy: {results["accurate"].mean()}')

import matplotlib.pyplot as plt
plt.scatter(results['label'], results['prob'])
plt.savefig(
    'verify_combined_model.png',
    dpi=300, bbox_inches='tight'
); plt.close()

