
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.dpath_dataRoot = Path('../data')
cfg.dpath_ckpt = Path('ckpt')
cfg.dpath_features = cfg.dpath_dataRoot / 'gliVmen_features'

cfg.dpath_features.mkdir(exist_ok=True)

cfg.fpath_manifest = Path('../mmMRI/_v6_datahandling/csv/manifest.csv')
cfg.fpath_log = Path('csv/log.csv')







