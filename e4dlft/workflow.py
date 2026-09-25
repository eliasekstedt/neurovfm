
import torch
import contextlib
import io

from e4dlft.preprocessor import StudyPreprocessor
from e4dlft.encoder import Encoder
from e4dlft.end2end import End2End, FCPart, End2EndForCaptum

def compare(encoder, device, modality, preproc, dpath_nii, fpath_FCStatedict):
    def get_standards():
        from neurovfm.pipelines.encoder import load_encoder
        encoder, preproc = load_encoder('ckpt')
        return encoder, preproc
    
    import pandas as pd
    std_encoder, std_preproc = get_standards()
    clsfr = FCPart(embed_dim=768, dropout=0.5)
    clsfr.load_state_dict(torch.load(fpath_FCStatedict))

    logits_lst = []
    for fx, proc in [[encoder, preproc], [std_encoder, std_preproc]]:
        e2e = End2End(fx, clsfr).to(device)
        e2e.eval()
        id = pd.read_csv(fpath_FCStatedict.parent / 'eval.csv')['id'].to_list()[0]
        fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            batch = proc(fpath_nii, 'mri')
        logits = e2e(batch)
        logits_lst.append(logits)

    for logits in logits_lst:
        print(logits.shape)
        print(logits[:5])
    print(torch.allclose(logits_lst[0], logits_lst[1]))

class Workflow:
    def __init__(self, device, modality, vit_params, dpath_nii, fpath_encoderStatedict, fpath_FCStatedict):
        preproc = StudyPreprocessor()
        #compare(encoder, device, modality, preproc, dpath_nii, fpath_FCStatedict)
        e2e = self.build_e2e(device, vit_params, fpath_encoderStatedict, fpath_FCStatedict)
        #self.readiness_test(e2e, modality, preproc, dpath_nii)
        self.run_captum(e2e, modality, preproc, dpath_nii)

    def run_captum(self, e2e, modality, preproc, dpath_nii):
        id = 'BraTS19_CBICA_AVJ_1'
        fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            batch = preproc(fpath_nii, 'mri')

        e2e4captum = End2EndForCaptum(e2e, batch)

        x = batch['img'].clone()
        baseline = torch.zeros_like(x)


        if False:
            from captum.attr import DeepLift
            explainer = DeepLift(e2e4captum)
            attr = explainer.attribute(
                inputs=x,
                baselines=baseline
            )
        if True:
            from captum.attr import IntegratedGradients
            explainer = IntegratedGradients(e2e4captum)
            attr = explainer.attribute(
                inputs=x,
                baselines=baseline,
            )

        print(attr.shape)
        print(attr.abs().sum())

    def build_e2e(self, device, vit_params, fpath_encoderStatedict, fpath_FCStatedict):
        encoder = Encoder(vit_params=vit_params, device=device, fpath_encoderStatedict=fpath_encoderStatedict)
        classifier = FCPart(embed_dim=768, dropout=0.5)
        classifier.load_state_dict(torch.load(fpath_FCStatedict))
        e2e = End2End(encoder, classifier).to(device)
        e2e.eval()
        return e2e
"""
    def readiness_test(self, e2e, modality, preproc, dpath_nii):
        def test_deeplift_ready(end2end, batch):
            x = batch["img"].clone().requires_grad_(True)
            assert x.requires_grad

            # Can autograd see the output?
            batch2 = dict(batch)
            batch2["img"] = x
            logit = end2end(batch2)
            assert logit.requires_grad
            assert logit.grad_fn is not None

            # Did gradients reach the input?
            logit.sum().backward()
            assert x.grad is not None
            assert x.grad.abs().sum().item() > 0
            print("DeepLIFT readiness test passed")

        id = 'BraTS19_CBICA_AVJ_1'
        fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'
        with contextlib.redirect_stdout(io.StringIO()):
            batch = preproc(fpath_nii, 'mri')




        test_deeplift_ready(e2e, batch)
"""




