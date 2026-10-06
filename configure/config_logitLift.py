
from types import SimpleNamespace
from pathlib import Path

def create_dirs(*dpaths):
    for dpath in dpaths:
        dpath.mkdir(exist_ok=True)

#id = 'BraTS19_CBICA_AVJ_1'

cfg = SimpleNamespace()
##########################
cfg.do_downFlow = False
cfg.do_encoderFlow = False
cfg.do_logitFlow = False
cfg.do_upFlow = True
##########################

cfg.dpath_dataRoot = Path('../data/logitLift')
cfg.dpath_niiRoot = cfg.dpath_dataRoot / 'brats19_LHg'
cfg.dpath_voxlvec = cfg.dpath_dataRoot / 'voxlvec'
cfg.dpath_embvec = cfg.dpath_dataRoot / 'embvec'
cfg.dpath_logitvec = cfg.dpath_dataRoot / 'logitvec'
cfg.dpath_rebuilt = cfg.dpath_dataRoot / 'rebuilt'

create_dirs(
    cfg.dpath_voxlvec,
    cfg.dpath_embvec,
    cfg.dpath_logitvec,
    cfg.dpath_rebuilt,
)

cfg.fpath_encoderState = Path('ckpt/pytorch_model.bin')
cfg.fpath_FCState = Path('one_run/model.pth')

cfg.mriseq = 't1ce'
cfg.device = 'cuda:0'
cfg.buildkey = 'patch_logits'


#cfg.fpath_nii = cfg.dpath_dataRoot / id / f'{id}_t1ce.nii.gz'