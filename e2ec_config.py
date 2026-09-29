
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.fpath_FCStatedict = Path('one_run/model.pth')
cfg.fpath_evalData = Path('one_run/eval.csv')
cfg.dpath_vec = Path('../data/brats19/features')

cfg.modality = 't1ce'

