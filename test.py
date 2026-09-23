
from config import *

def test0():
    import torch
    fpath = list(cfg.dpath_features.iterdir())[0]
    vec = torch.load(fpath)
    print(vec)
    print(vec.shape)
    print(vec.min(), vec.max())


def test1():
    from pathlib import Path
    fpath = Path('../mmMRI/_v5_NeuroVFM/run/report_better/22_07_36_00/model.pth')
    condition = True
    while condition:
        condition = not fpath.exists()
        print(condition, fpath)
        fpath = fpath.parent

test1()
"""
test0()
"""