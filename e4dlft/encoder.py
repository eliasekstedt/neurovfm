
import torch
from e4dlft.vit import VisionTransformer
from e4dlft.utils import NormalizationModule

def freeze(vit):
    for param in vit.parameters():
        param.requires_grad = False
    return vit

class Encoder:
    def __init__(self, vit_params, device, fpath_encoderStatedict):
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

