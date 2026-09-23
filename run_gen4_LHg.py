


import torch
import pandas as pd
from tqdm import tqdm
from neurovfm.pipelines.encoder import load_encoder

import contextlib
import io

class Wrapper:
    def __init__(self, modalities, dpath_ckpt, dpath_features, dpath_hgg, dpath_lgg, fpath_manifest):
        encoder, preproc = load_encoder(dpath_ckpt)
        
        if fpath_manifest.exists() and any(dpath_features.iterdir()):
            manifest = pd.read_csv(fpath_manifest)
        else:
            manifest = None


        fpaths_lgg = [item for item in list(dpath_lgg.iterdir()) if item.is_dir()]
        fpaths_hgg = [item for item in list(dpath_hgg.iterdir()) if item.is_dir()]
        lgg_items = list(zip(fpaths_lgg, [0] * len(fpaths_lgg)))
        hgg_items = list(zip(fpaths_hgg, [1] * len(fpaths_hgg)))
        data_items = lgg_items + hgg_items
        with torch.inference_mode():
            for dpath, label in tqdm(data_items):
                patient_id = dpath.name
                if not manifest is None and id in manifest['id'].to_list():
                    print(f'skipping {id}')
                    continue

                origin = dpath.parent.name
                dpath_this_nii = dpath_features / patient_id
                dpath_this_nii.mkdir(exist_ok=True)

                fpaths_nii = [
                    fpath for fpath in dpath.iterdir()
                    if any([fpath.name.endswith(f'_{mod}.nii.gz') for mod in modalities])
                    and not fpath.name.startswith('.')
                ]
                assert len(fpaths_nii) == 4

                for fpath_nii in fpaths_nii:
                    pt_name = f"{fpath_nii.name.removesuffix('.nii.gz')}.pt"
                    fpath_vec = dpath_this_nii / pt_name

                    with contextlib.redirect_stdout(io.StringIO()):
                        batch = preproc.load_study(
                            fpath_nii,
                            modality="mri",
                        )
                        vec = encoder.embed(batch)
                        vec = vec.mean(dim=0).detach().cpu().float()
                        torch.save({
                            'vec':vec,
                            'id':patient_id,
                            'modality':pt_name.split('_')[-1].removesuffix('.pt'),
                            'origin':origin,
                            'encoder_name':'nvfm',
                        }, fpath_vec)


                new_entry = pd.DataFrame({
                    'id':[patient_id],
                    'label':[label],
                    'origin':[origin],
                })

                if manifest is None:
                    manifest = new_entry
                else:
                    manifest = pd.concat([manifest, new_entry], axis=0)
                manifest.to_csv(fpath_manifest, index=False)


from config_LHg import *
Wrapper(
    modalities=cfg.modalities,
    dpath_ckpt=cfg.dpath_ckpt,
    dpath_features=cfg.dpath_features,
    dpath_hgg=cfg.dpath_hgg,
    dpath_lgg=cfg.dpath_lgg,
    fpath_manifest=cfg.fpath_manifest,
)



