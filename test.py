
from e4dlft_config import *

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


def test2():
    import pandas as pd
    id = pd.read_csv(cfg.fpath_FCStatedict.parent / 'eval.csv')['id'].to_list()[0]
    print(id)

test2()

"""
test1()
test0()
"""