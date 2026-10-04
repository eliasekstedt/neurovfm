
import torch
from tqdm import tqdm
from logitLift.down_process import DownProcess

class DownFlow:
    def __init__(self, mriseq, dpath_niiRoot, dpath_voxlvec):
        dpaths_nii = [
            dpath for dpath in dpath_niiRoot.iterdir()
            if not dpath.name.startswith('.')
        ]
        for dpath in tqdm(dpaths_nii):
            id = dpath.name
            fpath_nii = dpath / f"{id}_{mriseq}.nii.gz"
            batch = DownProcess(fpath_nii).batch
            batch['id'] = id
            batch['mriseq'] = mriseq
            fpath_voxlvec = dpath_voxlvec / f'{id}_{mriseq}.pt'
            #print(fpath_nii.exists(), fpath_nii)
            #print(fpath_voxlvec.exists(), fpath_voxlvec)
            #raise SystemExit
            torch.save(batch, fpath_voxlvec)
        print(batch.keys())



