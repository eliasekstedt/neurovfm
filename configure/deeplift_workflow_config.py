
from pathlib import Path
from types import SimpleNamespace

def create_dirs(*dpaths):
    for dpath in dpaths:
        dpath.mkdir(exist_ok=True)

cfg = SimpleNamespace()
#fpath_config = Path('cktp/config.json')
cfg.fpath_encoderStatedict = Path('ckpt/pytorch_model.bin')
cfg.fpath_FCStatedict = Path('one_run/model.pth')
cfg.fpath_evalData = Path('one_run/eval.csv')
cfg.dpath_nii = Path('../data/brats19/HGG')
cfg.dpath_attr = Path('../data/brats19_nvfm2deeplift_attr')

create_dirs(
    cfg.dpath_attr,
)

cfg.fpath_counter = None #Path('../data/brats19/LGG/BraTS19_TCIA10_410_1/BraTS19_TCIA10_410_1_t1ce.nii.gz')

cfg.modality = 't1ce'
cfg.ids = ['BraTS19_CBICA_AVJ_1', 'BraTS19_2013_14_1', 'BraTS19_CBICA_ATD_1']

cfg.vit_params = {
    "embed_dim":768,
    "depth":12,
    "num_heads":12,
    "prefix_len":0,
    "embed_layer_cf":{
        "which":"voxel",
        "params":{
            "in_chans":1,
            "embed_dim":738,
            "bias":True,
            "fused_bias_fc":True,
            "patch_hw_size":16,
            "patch_d_size":4,
        },
    },
    "pos_emb_cf":{
        "which":"pe3d",
        "params":{
            "in_dim":738,
            "d":30,
            "d_size":128,
            "hw_size":192,
            "pe_factor":1,
            "concat": True,
        },
    },
}

