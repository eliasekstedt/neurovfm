


import torch
import pandas as pd
from tqdm import tqdm
from neurovfm.pipelines.encoder import load_encoder

import contextlib
import io



class Wrapper:
    def __init__(self, fpath_log, fpath_manifest, dpath_dataRoot, dpath_features, dpath_ckpt):
        encoder, preproc = load_encoder(dpath_ckpt)
        
        if fpath_log.exists() and any(dpath_features.iterdir()):
            log = pd.read_csv(fpath_log)
        else:
            log = None


        manifest = pd.read_csv(fpath_manifest)
        with torch.inference_mode():
            for _, row in tqdm(manifest.iterrows(), total=manifest.shape[0]):
                id = row['id']
                if id in ['BraTS-MEN-RT-0692-1']:
                    print(f'bad id: {id}')
                    continue
                
                dataset_id = row['dataset']
                fname = row['filename']
                label = row['label']

                if not log is None and id in log['id'].to_list():
                    print(f"skipping {id}")
                    continue

                fpath_nii = dpath_dataRoot / dataset_id / id / fname
                if not fpath_nii.exists():
                    print(f'no file found for {fpath_nii}')
                    continue

                fpath_vec = dpath_features / f"{id}.pt"


                with contextlib.redirect_stdout(io.StringIO()):
                    batch = preproc.load_study(
                        fpath_nii,
                        modality="mri",
                    )

                vec = encoder.embed(batch)
                vec = vec.mean(dim=0).detach().cpu().float()
                
                torch.save({
                    'vec':vec,
                    'id':id,
                    'modality':'t1c_or_t1ce',
                    'encoder_name':'nvfm',
                }, fpath_vec)

                new_entry = pd.DataFrame({
                    'id':[id],
                    'label':[label],
                })

                if log is None:
                    log = new_entry
                else:
                    log = pd.concat([log, new_entry], axis=0)

                log.to_csv(fpath_log, index=False)




from config_gliVmen import *



Wrapper(
    fpath_log=cfg.fpath_log,
    fpath_manifest=cfg.fpath_manifest,
    dpath_dataRoot=cfg.dpath_dataRoot,
    dpath_features=cfg.dpath_features,
    dpath_ckpt=cfg.dpath_ckpt,
)



