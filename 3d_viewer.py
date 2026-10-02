
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import nibabel as nib


id = 'BraTS19_CBICA_AVJ_1'
fpath_nii = Path('../data/brats19_nvfm2deeplift_attr') / f"rebuilt_{id}.nii.gz"
record = nib.load(fpath_nii).get_fdata()

print(record.shape)
win_name = "rec"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

alive = True
while alive:
    for kk in range(record.shape[2]):
        canvas = record[:, :, kk]
        cv2.imshow(win_name, canvas)
        wkey = cv2.waitKey(10)

        if wkey == ord('Q') or wkey == ord('q') or wkey == 27:
            alive = False
            break
    record = record[:, :, ::-1]

cv2.destroyWindow(win_name)