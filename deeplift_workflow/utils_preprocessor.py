
import numpy as np
import SimpleITK as sitk
from typing import Tuple

#def load_image(fpath, preprocess=True):
def load_image(fpath, preprocess=True):
    assert fpath.name.endswith('.nii.gz')
    print(f"Loading NIfTI file: {fpath}")
    img_sitk = img_sitk = sitk.ReadImage(str(fpath))
    img_sitk = preprocess_image(img_sitk)
    print(f"Successfully loaded image: shape={img_sitk.GetSize()}, spacing={img_sitk.GetSpacing()}")
    return img_sitk

def preprocess_image(img_sitk):
    def reorient(img_sitk, tgt='RPI'):
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(tgt)
        return orienter.Execute(img_sitk)

    def compute_spacing(img_sitk):
        spacing_sitk = img_sitk.GetSpacing()
        return np.array(spacing_sitk, dtype=float)

    # Reorient to standard RPI coordinate system
    # RPI = Right-Posterior-Inferior (standard for medical image processing)
    img_sitk = reorient(img_sitk, tgt='RPI')  # (x, y, z)
    
    # Extract physical spacing information
    spacing_sitk = compute_spacing(img_sitk)  # -> np.array([x, y, z])
    
    original_spacing = spacing_sitk
    original_size = img_sitk.GetSize()
    
    # Determine slice dimension (z_dim) using heuristics
    # This identifies which dimension represents the slice/through-plane axis
    if len(set(list(original_spacing))) == 1:
        # All spacings equal - default to dimension 2
        z_dim = 2
    else:
        counts = np.bincount(original_size)
        if (counts == 1).all():
            # All sizes are unique - use dimension with largest spacing
            z_dim = np.argmax(original_spacing)
        else:
            # Use dimension with unique size (typically slice direction)
            z_dim = np.where(original_size == np.where(counts == 1)[0][0])[0][0]
    
    # Define target anisotropic spacing: 1x1x4 mm
    # In-plane: 1mm × 1mm for high resolution
    # Through-plane: 4mm for typical slice thickness
    target_spacing = [1, 1, 1]
    target_spacing[z_dim] = 4
    
    # Calculate new size based on spacing ratio
    new_size = [
        int(original_size[i] * (original_spacing[i] / target_spacing[i])) 
        for i in range(len(original_size))
    ]
    
    # Configure resampler with BSpline interpolation for smooth results
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    # Execute resampling
    resized_img_sitk = resampler.Execute(img_sitk)
    
    # Crop to dimensions divisible by patch sizes
    # In-plane: divisible by 16 for patch-based processing
    # Through-plane: divisible by 4 for slice-based processing
    start_index, crop_size = [], []
    for idx in range(3):
        if idx == z_dim:
            # Slice dimension: make divisible by 4
            start_index.append((new_size[idx] % 4) // 2)
            crop_size.append((new_size[idx] // 4) * 4)
        else:
            # In-plane dimensions: make divisible by 16
            start_index.append((new_size[idx] % 16) // 2)
            crop_size.append((new_size[idx] // 16) * 16)
    
    # Extract cropped region (centered crop)
    resized_img_sitk = sitk.Extract(resized_img_sitk, crop_size, start_index)
    
    return resized_img_sitk

def prepare_for_inference(img_sitk, mode, z_dim=None):
    def transpose_to_dhw(img_arr, z_dim):
        # SimpleITK returns array in (z, y, x) order
        # z_dim indicates which axis in (x, y, z) space is the slice dimension
        # We need to map this to the array indexing        
        # Convert z_dim from (x,y,z) to array (z,y,x) indexing
        view = 2 - z_dim
        
        # Transpose to put slice dimension first
        if view == 0:
            # Already in correct order [D, H, W]
            return img_arr, view
        elif view == 1:
            # Need to swap: (z, D, x) -> (D, z, x)
            return np.transpose(img_arr, (1, 0, 2)), view
        elif view == 2:
            # Need to swap: (z, y, D) -> (D, y, z)
            return np.transpose(img_arr, (2, 1, 0)), view
        else:
            # Default: no transpose
            return img_arr, view

    def get_background_mask_mri(img_arr):
        # MRI-specific background detection
        # Background is typically low intensity
        threshold = np.percentile(img_arr, 10)
        mask = img_arr > threshold
        return mask

    
    mode = mode.lower()
    assert mode == 'mri', f"Mode must be 'ct' or 'mri', got: {mode}"
    
    # Infer z_dim from spacing if not provided
    if z_dim is None:
        spacing = img_sitk.GetSpacing()
        if np.unique(spacing).shape[0] == 1:
            z_dim = 2 # assume axial
        else:
            z_dim = np.argmax(spacing)  # Dimension with largest spacing (4mm)
    
    # Convert SimpleITK image to numpy array
    # GetArrayFromImage returns (z, y, x) order
    img_arr = sitk.GetArrayFromImage(img_sitk)
    
    # Transpose to [D, H, W] where D is the slice dimension
    img_arr, view = transpose_to_dhw(img_arr, z_dim)
    D, H, W = img_arr.shape
    
    # Validate minimum dimensions
    if D < 4 or H < 16 or W < 16:
        print(f"Warning: Image too small (D={D}, H={H}, W={W}). Minimum: D>=4, H>=16, W>=16")
        return None
    
    # Ensure contiguous array in float64
    img_arr = img_arr.astype(np.float64).copy()
    
    # MRI: Percentile clipping and normalization
    background_mask = get_background_mask_mri(img_arr.copy())
    
    # Clip to 0.5th-99.5th percentile to remove outliers
    img_arr = np.clip(
        img_arr,
        np.percentile(img_arr, 0.5),
        np.percentile(img_arr, 99.5)
    )
    
    # Min-max normalization to [0, 1]
    img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
    
    img_arrs = [img_arr]
    return img_arrs, background_mask, view


def tokenize_volume(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    patch_size: Tuple[int, int, int] = (4, 16, 16),
    remove_background: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    import torch
    from einops import rearrange
    
    D, H, W = img_arr.shape
    p1, p2, p3 = patch_size
    
    # Validate dimensions are divisible by patch size
    assert D % p1 == 0, f"Depth {D} not divisible by patch size {p1}"
    assert H % p2 == 0, f"Height {H} not divisible by patch size {p2}"
    assert W % p3 == 0, f"Width {W} not divisible by patch size {p3}"
    
    # Calculate number of patches per dimension
    n_patches_d = D // p1
    n_patches_h = H // p2
    n_patches_w = W // p3
    
    # Tokenize image: [D, H, W] -> [N, patch_features]
    tokens_torch = rearrange(
        torch.from_numpy(img_arr).unsqueeze(0),  # Add channel dim [1, D, H, W]
        "c (d p1) (h p2) (w p3) -> (d h w) (c p1 p2 p3)",
        d=n_patches_d, h=n_patches_h, w=n_patches_w,
        p1=p1, p2=p2, p3=p3
    ).squeeze(1)  # Remove channel dim: [N, 1024]
    
    # Tokenize mask: a patch is background if ANY pixel in it is background
    # mask_arr is True for foreground, False for background
    mask_tokens = rearrange(
        torch.from_numpy(mask_arr),
        "(d p1) (h p2) (w p3) -> (d h w) (p1 p2 p3)",
        d=n_patches_d, h=n_patches_h, w=n_patches_w,
        p1=p1, p2=p2, p3=p3
    )
    # filtered: 1 if any pixel in patch is background (i.e., not all pixels are foreground)
    # 0 if all pixels in patch are foreground
    filtered = (~(mask_tokens.all(dim=1))).to(torch.uint8)  # [N]
    
    # Generate 3D coordinates for each token
    coords_d, coords_h, coords_w = np.meshgrid(
        np.arange(n_patches_d),
        np.arange(n_patches_h),
        np.arange(n_patches_w),
        indexing='ij'
    )
    coords_torch = torch.from_numpy(
        np.stack([coords_d.flatten(), coords_h.flatten(), coords_w.flatten()], axis=1)
    ).long()
    
    if remove_background:
        # Physically remove background tokens
        fg_mask = ~filtered.bool()
        tokens = tokens_torch[fg_mask].numpy()
        coords = coords_torch[fg_mask].numpy()
        filtered_out = np.array([], dtype=np.uint8)  # Empty since we removed them
    else:
        # Keep all tokens, return filtered mask
        tokens = tokens_torch.numpy()
        coords = coords_torch.numpy()
        filtered_out = filtered.numpy()
    
    return tokens, coords, filtered_out