
import SimpleITK as sitk
import numpy as np
import torch

class UpProcess:
    def __init__(self, batch, buildkey):
        self.buildkey = buildkey
        self.batch = batch
        self.rebuilt = self.reconstruct_original_geometry()
        
    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _inverse_transpose(arr_dhw, view):
        """
        Convert [D, H, W] back to the array order expected by SimpleITK:
        [z, y, x].
        """
        if view == 0:
            return arr_dhw
        elif view == 1:
            return np.transpose(arr_dhw, (1, 0, 2))
        elif view == 2:
            return np.transpose(arr_dhw, (2, 1, 0))
        else:
            raise ValueError(f"Invalid view: {view}")

    def reconstruct_processed_array(self, tokens=None, fill_value=0.0):
        """
        Reassemble flattened patches into the processed [D, H, W] volume.

        Missing background patches remain set to fill_value.
        """
        if tokens is None:
            tokens = self.batch[self.buildkey]

        tokens = self._to_numpy(tokens)
        coords = self._to_numpy(self.batch["coords"]).astype(np.int64)

        D, H, W = self.batch["size"][0]
        p1, p2, p3 = self.batch["patch_size"]

        expected_features = p1 * p2 * p3

        if tokens.ndim != 2:
            raise ValueError(
                f"Expected tokens with shape [N, features], got {tokens.shape}"
            )

        if tokens.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} values per token, "
                f"got {tokens.shape[1]}"
            )

        if len(tokens) != len(coords):
            raise ValueError(
                f"Token/coordinate mismatch: {len(tokens)} vs {len(coords)}"
            )

        volume = np.full(
            (D, H, W),
            fill_value,
            dtype=tokens.dtype,
        )

        for token, (d, h, w) in zip(tokens, coords):
            patch = token.reshape(p1, p2, p3)

            d0 = int(d) * p1
            h0 = int(h) * p2
            w0 = int(w) * p3

            volume[
                d0:d0 + p1,
                h0:h0 + p2,
                w0:w0 + p3,
            ] = patch

        return volume

    def reconstruct_processed_image(self, tokens=None, fill_value=0.0):
        """
        Return a SimpleITK image in the cropped/resampled NeuroVFM geometry.
        """
        volume_dhw = self.reconstruct_processed_array(
            tokens=tokens,
            fill_value=fill_value,
        )

        volume_zyx = self._inverse_transpose(
            volume_dhw,
            self.batch["view"],
        )

        reconstructed = sitk.GetImageFromArray(volume_zyx)

        processed_reference = self.batch["geometry"]["processed_img"]
        reconstructed.CopyInformation(processed_reference)

        return reconstructed

    def reconstruct_original_geometry(
        self,
        tokens=None,
        fill_value=0.0,
        interpolator=sitk.sitkLinear,
    ):
        """
        Return a SimpleITK image with the original NIfTI's:
        - size
        - spacing
        - origin
        - direction

        Information removed through cropping/background filtering remains zero.
        """
        processed_image = self.reconstruct_processed_image(
            tokens=tokens,
            fill_value=fill_value,
        )

        original_reference = self.batch["geometry"]["original_img"]

        restored = sitk.Resample(
            processed_image,
            original_reference,
            sitk.Transform(),
            interpolator,
            fill_value,
            processed_image.GetPixelID(),
        )

        return restored


