
from pathlib import Path
import cv2
import nibabel as nib

from configure.config_logitLift import *

id = 'BraTS19_CBICA_AOD_1'


fpath_nii = Path(cfg.dpath_rebuilt) / f"{id}.nii.gz"
record = nib.load(fpath_nii).get_fdata()
record = (record - record.min()) / (record.max() - record.min())
print(record)
print(record.shape)
#raise SystemExit

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
