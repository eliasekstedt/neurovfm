
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.dpath_dataRoot = Path('../data/brats24/')
cfg.dpath_patients = cfg.dpath_dataRoot / 'imgs'
cfg.dpath_features = cfg.dpath_dataRoot / 'nvfm_features_aggmean'
cfg.dpath_ckpt = Path('ckpt')
cfg.dpath_csv = Path('csv')

def create_dirs(*dpaths):
    for dpath in dpaths:
        dpath.mkdir(exist_ok=True)

create_dirs(
    cfg.dpath_features,
    cfg.dpath_csv,
)

cfg.fpath_log = cfg.dpath_csv / 'log.csv'

cfg.modalities = ['t1c', 't1n', 't2f', 't2w']


