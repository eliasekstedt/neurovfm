
from e4dlft_config import *

from e4dlft.workflow import Workflow
Workflow(
    device='cuda:0',
    modality=cfg.modality,
    vit_params=cfg.vit_params,
    dpath_nii=cfg.dpath_nii,
    fpath_encoderStatedict=cfg.fpath_encoderStatedict,
    fpath_FCStatedict=cfg.fpath_FCStatedict,
)

"""
figure out exactly how NeuroVFM uses the other values in
a batch; coords, series_cu_seqlens, etc
"""