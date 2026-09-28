
from e2ec_config import *

from e2ec.workflow import Workflow
Workflow(
    device='cuda:0',
    modality=cfg.modality,
    vit_params=cfg.vit_params,
    dpath_nii=cfg.dpath_nii,
    fpath_encoderStatedict=cfg.fpath_encoderStatedict,
    fpath_FCStatedict=cfg.fpath_FCStatedict,
)

