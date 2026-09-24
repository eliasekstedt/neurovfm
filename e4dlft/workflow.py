
import torch
import contextlib
import io

from e4dlft.preprocessor import StudyPreprocessor
from e4dlft.encoder import Encoder
from e4dlft.end2end import End2End, FCPart

def compare(encoder, device, modality, preproc, dpath_nii, fpath_FCStatedict):
    def get_standards():
        from neurovfm.pipelines.encoder import load_encoder
        encoder, preproc = load_encoder('ckpt')
        return encoder, preproc
    
    import pandas as pd
    std_encoder, std_preproc = get_standards()
    clsfr = FCPart(embed_dim=768, dropout=0.5)
    clsfr.load_state_dict(torch.load(fpath_FCStatedict))

    logits_lst = []
    for fx, proc in [[encoder, preproc], [std_encoder, std_preproc]]:
        e2e = End2End(fx, clsfr).to(device)
        e2e.eval()
        id = pd.read_csv(fpath_FCStatedict.parent / 'eval.csv')['id'].to_list()[0]
        fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            batch = proc(fpath_nii, 'mri')
        logits = e2e(batch)
        logits_lst.append(logits)
        continue

    for logits in logits_lst:
        print(logits.shape)
        print(logits[:5])
    print(torch.allclose(logits_lst[0], logits_lst[1]))

class Workflow:
    def __init__(self, device, modality, vit_params, dpath_nii, fpath_encoderStatedict, fpath_FCStatedict):
        preproc = StudyPreprocessor()
        encoder = Encoder(
            vit_params=vit_params,
            device=device,
            fpath_encoderStatedict=fpath_encoderStatedict,
        )

        compare(encoder, device, modality, preproc, dpath_nii, fpath_FCStatedict)

