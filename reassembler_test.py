
from pathlib import Path

from e2ec.utils_preprocessor import load_image, prepare_for_inference, tokenize_volume

import torch
import numpy as np



def detokenize_volume(tokens, coords, volume_shape, patch_size=(4, 16, 16)):
    D, H, W = volume_shape
    p1, p2, p3 = patch_size

    volume = np.zeros((D, H, W), dtype=tokens.dtype)

    for token, (d_idx, h_idx, w_idx) in zip(tokens, coords):
        patch = token.reshape(p1, p2, p3)

        d0 = d_idx * p1
        h0 = h_idx * p2
        w0 = w_idx * p3

        volume[d0:d0+p1, h0:h0+p2, w0:w0+p3] = patch

    return volume


def test2():
    def nii2tok(fpath_nii):
        img_sitk = load_image(fpath_nii, preprocess=True)
        img_arrs, background_mask, view = prepare_for_inference(img_sitk, 'mri')
        img_arr = img_arrs[0]
        tokens, coords, filtered = tokenize_volume(
            img_arr,
            background_mask,
            patch_size=(4, 16, 16),
            remove_background=True,
        )
        return tokens, coords
        
    dpath_nii = Path('../data/brats19/HGG')
    dpath_ptok = Path('../data/brats19_deeplift_attr_examples')
    dpath_fullTok = Path('../data/token_test')
    dpath_fullTok.mkdir(exist_ok=True)

    ids = ['BraTS19_CBICA_AVJ_1', 'BraTS19_2013_14_1', 'BraTS19_CBICA_ATD_1']
    for id in ids:
        fpath_nii = dpath_nii / id / f'{id}_t1ce.nii.gz'
        fpath_ptok = dpath_ptok / f'HGG_{id}.pt'
        rtok, coords = nii2tok(fpath_nii)
        attr = torch.load(fpath_ptok, weights_only=False).detach().numpy()
        torch.save({
            'tok_voxl':rtok,
            'tok_attr':attr,
            'coords':coords,
        }, dpath_fullTok / f'{id}.pt')

test2()
"""
"""

#import nibabel as nib
#img = nib.Nifti1Image(rebuilt, affine=np.eye(4))
#nib.save(img, Path('./rearray.nii'))