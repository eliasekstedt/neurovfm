from deeplift_workflow_config import *

from deeplift_workflow.workflow import Workflow
Workflow(
    ids=cfg.ids,
    device='cuda:0',
    modality=cfg.modality,
    vit_params=cfg.vit_params,
    dpath_nii=cfg.dpath_nii,
    dpath_attr=cfg.dpath_attr,
    fpath_encoderStatedict=cfg.fpath_encoderStatedict,
    fpath_FCStatedict=cfg.fpath_FCStatedict,
    fpath_counter=cfg.fpath_counter,
)

"""


how can the classifier work when each element in the input vector
is the mean of the same element from each patch?


"""