
from pathlib import Path
from types import SimpleNamespace

def create_dirs(*dpaths):
    for dpath in dpaths:
        dpath.mkdir(exist_ok=True)

cfg = SimpleNamespace()
cfg.dpath_dataRoot = Path('../data/logitLift/brats19_LHg')
