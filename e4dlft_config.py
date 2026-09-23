
from pathlib import Path
from types import SimpleNamespace

cfg = SimpleNamespace()
#fpath_config = Path('cktp/config.json')
cfg.fpath_encoderStatedict = Path('ckpt/pytorch_model.bin')
cfg.fpath_FCStatedict = Path('run/22_07_36_00/model.pth')

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

