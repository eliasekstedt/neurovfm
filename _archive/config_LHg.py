
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.dpath_dataRoot = Path('../data/brats19')
cfg.dpath_lgg = cfg.dpath_dataRoot / 'LGG'
cfg.dpath_hgg = cfg.dpath_dataRoot / 'HGG'
cfg.dpath_features = cfg.dpath_dataRoot / 'features'
cfg.dpath_ckpt = Path('ckpt')
cfg.dpath_csv = Path('csv/LHg')

def create_dirs(*dpaths):
    for dpath in dpaths:
        dpath.mkdir(exist_ok=True)

create_dirs(
    cfg.dpath_features,
    cfg.dpath_csv,
)

cfg.fpath_manifest = cfg.dpath_csv / 'manifest.csv'

cfg.modalities = ['flair', 't1', 't1ce', 't2']


