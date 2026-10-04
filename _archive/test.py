
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

def test3():
    from pathlib import Path
    id = 'BraTS19_CBICA_AVJ_1'
    dpath_nii = Path('../data/brats19/HGG/')
    assert dpath_nii.exists()
    modality = 't1ce'
    fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'

    import nibabel as nib
    t = nib.load(fpath_nii)
    print(t.shape)
    print(240*240*155)
    print(208*1024)
    print(208*1024/(240*240*155))

def test4():
    from pathlib import Path
    import contextlib
    import io
    from e2ec.preprocessor import StudyPreprocessor
    preproc = StudyPreprocessor()
    dpath_nii = Path('../data/brats19/HGG')
    ids = [
        'BraTS19_CBICA_AVJ_1',
        'BraTS19_2013_14_1',
        'BraTS19_CBICA_ATD_1',
    ]

    modality = 't1ce'
    for id in ids:
        fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'
        #with contextlib.redirect_stdout(io.StringIO()):
        batch = preproc(fpath_nii, 'mri')
        raise SystemExit

def test5():
    dpath_nii = Path('../data/brats19/HGG')
    patch_size = (4, 16, 16)
    target_spacing = (1.0, 1.0, 4.0)
    remove_background = True

    ids = ['BraTS19_CBICA_AVJ_1', 'BraTS19_2013_14_1', 'BraTS19_CBICA_ATD_1']
    fpaths_nii = [dpath_nii / id / f'{id}_t1ce.nii.gz' for id in ids]
    for fpath_nii in fpaths_nii:
        img_sitk = load_image(fpath_nii, preprocess=True)
        img_arrs, background_mask, view = prepare_for_inference(img_sitk, 'mri')
        img_arr = img_arrs[0]
        tokens, coords, filtered = tokenize_volume(
            img_arr,
            background_mask,
            patch_size=patch_size,
            remove_background=remove_background,
        )
        continue

def test6():
    ids = ['BraTS19_CBICA_AVJ_1', 'BraTS19_2013_14_1', 'BraTS19_CBICA_ATD_1']
    dpath_tok = Path('../data/brats19_deeplift_attr_examples')
    patch_size = (4, 16, 16)
    vol_dims = (36, 240, 240)
    eps = 1e-8
    for id in ids:
        fpath_tok = dpath_tok / f'HGG_{id}.pt'
        tok = torch.load(fpath_tok, weights_only=False).detach().numpy()
        tok = (tok - tok.min()) / (tok.max() - tok.min() + eps)
        rebuilt = detokenize_volume(tok, coords, vol_dims, patch_size)
        continue


"""
test4()
test3()
test2()
test1()
test0()
"""