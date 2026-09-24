
import torch
import pandas as pd
from tqdm import tqdm
from neurovfm.pipelines.encoder import load_encoder

import contextlib
import io



class Wrapper:
    def __init__(self, fpath_log, modalities, dpath_patients, dpath_features, dpath_ckpt):
        encoder, preproc = load_encoder(dpath_ckpt)
        
        for mod in modalities:
            self.gen_features_on_modality(mod, encoder, preproc, dpath_patients, dpath_features, fpath_log)
            

    def gen_features_on_modality(self, mod, encoder, preproc, dpath_patients, dpath_features, fpath_log):
        if fpath_log.exists() and any(dpath_features.iterdir()):
            log = pd.read_csv(fpath_log)
        else:
            log = None

        patients_list = list(dpath_patients.iterdir())
        with torch.inference_mode():
            for dpath_patient in tqdm(patients_list):
                scan_id = f"{dpath_patient.name}-{mod}"
                if not log is None and scan_id in log['scan_id'].to_list():
                    print(f"skipping {scan_id}")
                    continue

                fpath_nii = dpath_patient / f"{scan_id}.nii.gz"
                fpath_vec = dpath_features / f"{scan_id}.pt"

                with contextlib.redirect_stdout(io.StringIO()):
                    batch = preproc.load_study(
                        fpath_nii,
                        modality="mri",
                    )

                vec = encoder.embed(batch)
                vec = vec.mean(dim=0).detach().cpu().float()
                
                torch.save({
                    'vec':vec,
                    'scan_id':scan_id,
                    'modality':mod,
                    'encoder_name':'nvfm',
                }, fpath_vec)

                new_entry = pd.DataFrame({'scan_id':[scan_id]})
                if log is None:
                    log = new_entry
                else:
                    log = pd.concat([log, new_entry], axis=0)
                log.to_csv(fpath_log, index=False)




from config import *
Wrapper(
    fpath_log=cfg.fpath_log,
    modalities=cfg.modalities,
    dpath_patients=cfg.dpath_patients,
    dpath_features=cfg.dpath_features,
    dpath_ckpt=cfg.dpath_ckpt,
)


