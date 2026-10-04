
import torch
from tqdm import tqdm

from logitLift.encoder import Encoder

class EncoderFlow:
    def __init__(self, device, dpath_voxlvec, dpath_embvec, fpath_encoderState):
        encoder = Encoder(device, fpath_encoderState)
        for fpath in tqdm(list(dpath_voxlvec.iterdir())):
            batch = torch.load(fpath, weights_only=False)
            id = batch['id']
            mriseq = batch['mriseq']
            batch['embvec'] = encoder.embed(batch)
            fpath_embvec = dpath_embvec / f'{id}_{mriseq}.pt'
            torch.save(batch, fpath_embvec)
