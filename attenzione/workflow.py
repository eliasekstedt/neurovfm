
import torch
import SimpleITK as sitk
import contextlib
import io

from attenzione.encoder import Encoder
from attenzione.end2end import End2End, FCPart, ModelWrapper


class Workflow:
    def __init__(self, ids, device, modality, vit_params, dpath_nii, dpath_attr, fpath_encoderStatedict, fpath_FCStatedict, fpath_counter):
        model = self.build_model(device, vit_params, fpath_encoderStatedict, fpath_FCStatedict)
        for id in ids:
            fpath_nii = dpath_nii / id / f'{id}_{modality}.nii.gz'
            meta_for_deeplift = self.get_meta_for_deeplift(fpath_nii)
            meta = self.run_captum(model, meta_for_deeplift, fpath_nii, fpath_counter)
            up_process = UpProcess(meta)
            rebuilt = up_process.reconstruct_original_geometry()
            sitk.WriteImage(rebuilt, dpath_attr / f'rebuilt_{id}.nii.gz')

    def get_meta_for_deeplift(self, fpath_nii):
        from deeplift_workflow.preprocessor import StudyPreprocessor
        preproc = StudyPreprocessor(remove_background=False)
        with contextlib.redirect_stdout(io.StringIO()):
            meta_for_deeplift = preproc([fpath_nii]*2, 'mri')
        del meta_for_deeplift['img']
        return meta_for_deeplift

    def run_captum(self, model, meta_for_deeplift, fpath_nii, fpath_counter):
        def build_batches(fpath_nii, fpath_counter):
            def get_meta(batch):
                meta = dict(batch)
                del meta['img']
                return meta
            
            a_batch = DownProcess(fpath_nii).batch
            meta = get_meta(a_batch)
            b_batch = dict(meta)
            if fpath_counter is None:
                b_batch['img'] = torch.zeros_like(a_batch['img'])
            else:
                b_batch['img'] = DownProcess(fpath_counter).batch['img']
            return a_batch, b_batch, meta


        a_batch, b_batch, meta = build_batches(fpath_nii, fpath_counter)
        model_wrapper = ModelWrapper(model, meta_for_deeplift)

        from captum.attr import DeepLift
        explainer = DeepLift(model_wrapper)
        attr = explainer.attribute(
            inputs=a_batch['img'],
            baselines=b_batch['img'],
        )
        attr = torch.where(attr == 0, attr.min(), attr)
        attr = (attr - attr.min()) / (attr.max() - attr.min())

        meta['img'] = attr
        return meta

    def build_model(self, device, vit_params, fpath_encoderStatedict, fpath_FCStatedict):
        encoder = Encoder(vit_params=vit_params, device=device, fpath_encoderStatedict=fpath_encoderStatedict)
        classifier = FCPart(embed_dim=768, dropout=0.5)
        classifier.load_state_dict(torch.load(fpath_FCStatedict))
        model = End2End(encoder, classifier).to(device)
        model.eval()
        return model




