
import torch

from e2ec.end2end import ANN


class Workflow:
    def __init__(self, device, modality, dpath_vec, fpath_FCStatedict):
        assembly = self.assemble(device, fpath_FCStatedict)
        self.run_captum(assembly, modality, device, dpath_vec)

    def run_captum(self, assembly, modality, device, dpath_vec):
        id = 'BraTS19_2013_0_1'
        fpath_vec = dpath_vec / id / f'{id}_{modality}.pt'
        batch = torch.load(fpath_vec)
        x = batch['vec'].clone().reshape(1, 768).to(device)
        baseline = torch.zeros_like(x)

        from captum.attr import DeepLift
        explainer = DeepLift(assembly)
        attr = explainer.attribute(
            inputs=x,
            baselines=baseline,
        ).cpu().squeeze(0)


        values, idx = torch.topk(attr, k=5)
        print(values, idx)
        values, idx = torch.topk(attr, k=5, largest=False)
        print(values, idx)

    def assemble(self, device, fpath_FCStatedict):
        architecture = ANN(embed_dim=768, dropout=0.5)
        architecture.load_state_dict(torch.load(fpath_FCStatedict))
        architecture = architecture.to(device)
        architecture.eval()
        return architecture
