
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace()
cfg.fpath_FCStatedict = Path('one_run/model.pth')
cfg.fpath_evalData = Path('one_run/eval.csv')
cfg.dpath_nii = Path('../data/brats19/HGG')

cfg.modality = 't1ce'

