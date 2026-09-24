
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


device = 'cuda:0'

from neurovfm.pipelines.encoder import load_encoder
encoder, preproc = load_encoder("ckpt")


fpath_ann = Path('../mmMRI/_v5_NeuroVFM/run/report_better/22_07_36_00/model.pth')
fpath_evaldata = Path('../mmMRI/_v5_NeuroVFM/run/report_better/22_07_36_00/eval.csv')
ann = Model(embed_dim=768, dropout=0.5)
ann.load_state_dict(torch.load(fpath_ann))

model = CombinedModel(
    encoder=encoder,
    classifier=ann,
).to(device)
model.eval()

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

            ###
            print("==== BATCH KEYS ====")
            print(batch.keys())

            print("\n==== IMG ====")
            print(batch["img"].shape)
            print(batch["img"].dtype)
            print(batch["img"].requires_grad)

            print("\n==== COORDS ====")
            print(batch["coords"].shape)
            print(batch["coords"].dtype)
            print(batch["coords"][:10])

            print("\n==== SERIES ====")
            print(batch["series_cu_seqlens"])
            print(batch["series_max_len"])

            print("\n==== STUDY ====")
            print(batch["study_cu_seqlens"])
            print(batch["study_max_len"])

            print("\n==== MISC ====")
            print("mode:", batch["mode"])
            print("path:", batch["path"])
            print("size:", batch["size"])

            print("\n==== EMBEDDINGS ====")
            tokens = encoder.embed(batch)
            print(tokens.shape)
            print(tokens.dtype)

            vec = tokens.mean(dim=0)
            print("pooled:", vec.shape, vec.dtype)
            raise SystemExit
            ###
        
            logits = model(batch)
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

