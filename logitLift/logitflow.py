

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

class LogitFlow:
    def __init__(self, device, dpath_embvec, dpath_logitvec, fpath_FCState):
        embed_dim = 768
        model = self.init_model(fpath_FCState, embed_dim, device)
        for fpath in tqdm(list(dpath_embvec.iterdir())):
            batch = torch.load(fpath, weights_only=False)
            id = batch['id']
            loader = self.init_loader(fpath)
            logits = self.gen_logits(loader, model, device)
            batch['patch_logits'] = logits.repeat(batch['img'].shape[0], 1)
            fpath_logitvec = dpath_logitvec / f"{id}.pt"
            torch.save(batch, fpath_logitvec)

    def init_loader(self, tokens):
        from logitLift.reader import Reader
        return DataLoader(Reader(tokens), int(1e4), shuffle=False)

    def init_model(self, fpath_FCState, embed_dim, device):
        print('initiating model ...')
        from logitLift.model import FCPart
        model = FCPart(embed_dim, 0.0)
        model.load_state_dict(torch.load(fpath_FCState, map_location='cuda:0'))
        return model.to(device)

    def gen_logits(self, loader, model, device):
        model.eval()
        with torch.inference_mode():
            for idx, x in loader:
                x = x.to(device)
                logits = model(x)
        return logits.squeeze(1).cpu()
        
class UpFlow:
    """
    workflow for logited Nx768 imgs to nii
    """
