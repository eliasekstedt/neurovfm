
import torch
from attenzione.vit import VisionTransformer
from attenzione.utils import NormalizationModule

def freeze(vit):
    for param in vit.parameters():
        param.requires_grad = False
    return vit

class Encoder:
    def __init__(self, device, fpath_encoderStatedict):
        vit_params = {
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
        vit = self.load_vit(vit_params, fpath_encoderStatedict)
        normstats = [
            [0.3141, 0.4139, 0.3184, 0.2719],  # means: [mri, brain, blood, bone]
            [0.2623, 0.4059, 0.3605, 0.1875],   # stds
        ]
        self.device = device
        self.norm_module = NormalizationModule(custom_stats_list=normstats).to(self.device)
        self.vit = vit.to(self.device)
        self.vit = freeze(vit)

    def load_vit(self, vit_params, fpath_encoderStatedict):
        vit = VisionTransformer(**vit_params)
        state_dict = torch.load(fpath_encoderStatedict, map_location="cpu")
        if "state_dict" in state_dict:
            print('entered if')
            state_dict = state_dict["state_dict"]
        vit.load_state_dict(state_dict, strict=False)
        return vit

    def embed(self, batch):
        tokens = batch["img"].to(self.device)
        coords = batch["coords"].to(self.device)
        series_cu_seqlens = batch["series_cu_seqlens"].to(self.device)
        series_max_len = batch["series_max_len"]

        if batch.get("series_masks_indices") is not None and batch["series_masks_indices"].numel() > 0:
            print('bg not removed')#; raise SystemExit
            masks = batch["series_masks_indices"].to(self.device)
        else:
            print('bg was removed')#; raise SystemExit
            masks = None  # Background already removed

        tokens = self.norm_module.normalize(
            tokens,
            batch["mode"],
            batch["path"],
            cu_seqlens=series_cu_seqlens,
            sizes=batch.get("size")
        )

        amp_dtype = torch.bfloat16 if True else torch.float32
        with torch.amp.autocast(device_type=self.device, dtype=amp_dtype):
            embs = self.vit(
                tokens,
                coords,
                masks=masks,
                cu_seqlens=series_cu_seqlens,
                max_seqlen=series_max_len,
                use_flash_attn=False  # Disable for inference
            )
        
        return embs