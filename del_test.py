import torch
import SimpleITK as sitk
from tqdm import tqdm
from logitLift.up_process import UpProcess

class UpFlow:
    def __init__(self, buildkey, dpath_logitvec, dpath_rebuilt):
        for fpath in tqdm(list(dpath_logitvec.iterdir())):
            batch = torch.load(fpath, weights_only=False)
            print(batch['img'].shape)
            print(batch['patch_logits'].shape)
            raise SystemExit
            rebuilt = UpProcess(batch, buildkey).rebuilt
            
            fpath_rebuilt = dpath_rebuilt / f"{batch['id']}.nii.gz"
            sitk.WriteImage(rebuilt, fpath_rebuilt)


"""
from config_logitLift import cfg
UpFlow(
    buildkey=cfg.buildkey,
    dpath_logitvec=cfg.dpath_logitvec,
    dpath_rebuilt=cfg.dpath_rebuilt,
)
"""

import torch
img = torch.tensor([[0] * 5]*3)
print(img)
print(img.shape)
logits = torch.tensor([1, 2, 3, 4, 5])
recon = logits.repeat(img.shape[0], 1)
print('---')
print(recon)