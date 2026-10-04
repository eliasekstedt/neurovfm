
import torch
import SimpleITK as sitk
from tqdm import tqdm
from logitLift.up_process import UpProcess

class UpFlow:
    def __init__(self, buildkey, dpath_logitvec, dpath_rebuilt):
        for fpath in tqdm(list(dpath_logitvec.iterdir())):
            batch = torch.load(fpath, weights_only=False)
            rebuilt = UpProcess(batch, buildkey).rebuilt
            
            fpath_rebuilt = dpath_rebuilt / f"{batch['id']}.nii.gz"
            sitk.WriteImage(rebuilt, fpath_rebuilt)
            raise SystemExit