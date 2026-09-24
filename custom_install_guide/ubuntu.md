# NeuroVFM installation on Ubuntu 24.04

This guide reproduces the working NeuroVFM environment tested on the following system:

```text
Ubuntu:            24.04
GPU:               NVIDIA GeForce GTX 1060 6GB
NVIDIA driver:     580.173.02
Python:            3.10.x
PyTorch:           2.5.0+cu124
CUDA Toolkit:      12.5
FlashAttention:    2.6.3
torch-scatter:     2.1.2
NeuroVFM:          0.1.0
```

## Important distinctions

Three CUDA-related components are involved:

1. **NVIDIA driver**  
   Makes the GPU accessible to the operating system and PyTorch.

2. **PyTorch CUDA runtime**  
   Installed with the `cu124` PyTorch wheel and used when PyTorch runs CUDA operations.

3. **System CUDA Toolkit**  
   Provides `nvcc`, CUDA headers, and development libraries needed to compile FlashAttention, `fused_dense_lib`, and `torch-scatter`.

On this system, PyTorch uses CUDA 12.4 while the system compiler comes from CUDA Toolkit 12.5. PyTorch reports this as a minor-version mismatch, but all required extensions compiled and loaded successfully.

---

# 1. Check the starting system

This verifies that the NVIDIA driver sees the GPU and records the compiler and operating-system versions.

```bash
nvidia-smi && \
echo && \
echo "===== NVCC =====" && \
which nvcc || true && \
nvcc --version || true && \
echo && \
echo "===== COMPILER =====" && \
gcc --version | head -n 1 && \
g++ --version | head -n 1 && \
echo && \
echo "===== OPERATING SYSTEM =====" && \
lsb_release -a
```

It is normal for `nvidia-smi` to work even when `nvcc` is not installed.

The CUDA version displayed by `nvidia-smi` describes driver compatibility. It does not confirm that the CUDA Toolkit is installed.

---

# 2. Create a dedicated conda environment

This isolates NeuroVFM and its compiled dependencies from the base environment and other projects.

```bash
conda create -p ~/conda-envs/nvfm python=3.10 -y && \
conda activate ~/conda-envs/nvfm && \
which python && \
python --version
```

Expected Python path:

```text
/home/elias/conda-envs/nvfm/bin/python
```

Before running later installation commands, verify that the terminal prompt shows the `nvfm` environment rather than `(base)`.

---

# 3. Install PyTorch 2.5.0 with CUDA 12.4

This reproduces the PyTorch build known to work with NeuroVFM and FlashAttention 2.6.3.

```bash
python -m pip install \
  torch==2.5.0 \
  torchvision==0.20.0 \
  torchaudio==2.5.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

Verify PyTorch and GPU access:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Expected output includes:

```text
torch: 2.5.0+cu124
torch CUDA: 12.4
CUDA available: True
```

The PyTorch wheel contains the CUDA runtime libraries required to run PyTorch. It does not provide the full CUDA Toolkit or `nvcc` required to compile extensions.

---

# 4. Add NVIDIA's Ubuntu CUDA repository

Ubuntu's standard repository may not contain the required CUDA Toolkit packages. Add NVIDIA's package repository for Ubuntu 24.04.

```bash
cd /tmp && \
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
sudo apt update
```

Confirm that the CUDA 12.5 toolkit is available:

```bash
apt-cache policy cuda-toolkit-12-5 && \
apt-cache policy cuda-nvcc-12-5
```

---

# 5. Install the full CUDA 12.5 Toolkit

Install the full toolkit rather than only `cuda-nvcc-12-5`.

Installing only `nvcc` was insufficient because FlashAttention also required CUDA development headers such as `cusparse.h`.

```bash
sudo apt install -y cuda-toolkit-12-5
```

Configure CUDA for the current shell:

```bash
export CUDA_HOME=/usr/local/cuda-12.5
export PATH="$CUDA_HOME/bin:$PATH"
```

Verify the CUDA compiler and required headers:

```bash
echo "===== CUDA COMPILER =====" && \
which nvcc && \
nvcc --version && \
echo && \
echo "===== REQUIRED HEADER =====" && \
ls -l "$CUDA_HOME/include/cusparse.h" && \
echo && \
echo "===== PYTORCH CUDA CONFIGURATION =====" && \
python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA_HOME:", CUDA_HOME)
PY
```

Expected values:

```text
nvcc: CUDA 12.5
torch CUDA: 12.4
CUDA_HOME: /usr/local/cuda-12.5
```

---

# 6. Persist the CUDA configuration

Add the CUDA configuration to `~/.bashrc` without adding duplicate entries:

```bash
grep -qxF 'export CUDA_HOME=/usr/local/cuda-12.5' ~/.bashrc || \
echo 'export CUDA_HOME=/usr/local/cuda-12.5' >> ~/.bashrc

grep -qxF 'export PATH=$CUDA_HOME/bin:$PATH' ~/.bashrc || \
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
```

For the current terminal, retain:

```bash
export CUDA_HOME=/usr/local/cuda-12.5
export PATH="$CUDA_HOME/bin:$PATH"
```

If `source ~/.bashrc` returns the terminal to `(base)`, reactivate the environment:

```bash
conda activate ~/conda-envs/nvfm
```

---

# 7. Install FlashAttention build dependencies

FlashAttention requires Python build packages and uses Ninja to compile its C++ and CUDA sources.

```bash
python -m pip install \
  psutil \
  packaging \
  ninja \
  wheel \
  setuptools
```

Verify Ninja:

```bash
ninja --version
```

---

# 8. Build and install FlashAttention 2.6.3

Export the CUDA configuration and install FlashAttention without build isolation:

```bash
export CUDA_HOME=/usr/local/cuda-12.5 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

This may compile FlashAttention from source and can remain at the following message for a considerable period:

```text
Building wheel for flash-attn ... still running
```

To verify progress from another terminal, run:

```bash
top
```

Processes such as the following indicate that compilation is active:

```text
ninja
nvcc
ptxas
c++
cc1plus
```

A successful build ends with:

```text
Successfully built flash-attn
Successfully installed flash-attn-2.6.3
```

Verify the installation:

```bash
python - <<'PY'
import flash_attn

print("flash-attn:", flash_attn.__version__)
PY
```

Expected:

```text
flash-attn: 2.6.3
```

---

# 9. Build the separate fused_dense_lib extension

FlashAttention 2.6.3 does not install `fused_dense_lib` as part of its main package installation. NeuroVFM requires this extension through `flash_attn.ops.fused_dense.FusedDense`.

Clone the matching FlashAttention release:

```bash
cd /mnt/work/repo && \
git clone --branch v2.6.3 --depth 1 \
  https://github.com/Dao-AILab/flash-attention.git \
  flash-attention-2.6.3
```

Build and install `fused_dense_lib`:

```bash
cd /mnt/work/repo/flash-attention-2.6.3/csrc/fused_dense_lib && \
export CUDA_HOME=/usr/local/cuda-12.5 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
python -m pip install -v --no-build-isolation .
```

The `--no-build-isolation` option is necessary because the extension's build system imports PyTorch. An isolated pip build environment cannot see the PyTorch package installed in the `nvfm` environment.

Verify the extension:

```bash
python - <<'PY'
from flash_attn.ops.fused_dense import FusedDense

print("FusedDense OK")
PY
```

Expected:

```text
FusedDense OK
```

---

# 10. Clone the NeuroVFM repository

Replace `YOUR_NEUROVFM_REPOSITORY_URL` with the URL of the required fork.

```bash
cd /mnt/work/repo && \
git clone YOUR_NEUROVFM_REPOSITORY_URL neurovfm && \
cd neurovfm
```

If a `neurovfm` directory already exists, move it before cloning:

```bash
cd /mnt/work/repo && \
mv neurovfm neurovfm_backup && \
git clone YOUR_NEUROVFM_REPOSITORY_URL neurovfm && \
cd neurovfm
```

The backup can be removed after the replacement repository has been installed and tested successfully.

---

# 11. Relax the exact Python patch-version requirement

The original `pyproject.toml` requires exactly Python 3.10.14:

```toml
requires-python = "==3.10.14"
```

This rejects other Python 3.10 patch releases, including Python 3.10.21.

Inspect the current constraint:

```bash
cd /mnt/work/repo/neurovfm && \
grep -n "requires-python" pyproject.toml
```

Replace the exact patch-version requirement with a Python 3.10 range:

```bash
sed -i \
  's/requires-python = "==3\.10\.14"/requires-python = ">=3.10,<3.11"/' \
  pyproject.toml && \
grep -n "requires-python" pyproject.toml
```

Expected:

```toml
requires-python = ">=3.10,<3.11"
```

Ensure that the file contains literal `<` and `>` characters, not HTML entities such as `&lt;` and `&gt;`.

This change should preferably be committed to the fork.

---

# 12. Install torch-scatter separately

NeuroVFM requires `torch-scatter==2.1.2`.

Installing NeuroVFM before installing `torch-scatter` caused pip to build `torch-scatter` in an isolated environment, resulting in:

```text
ModuleNotFoundError: No module named 'torch'
```

Install it separately without build isolation:

```bash
cd /mnt/work/repo/neurovfm && \
export CUDA_HOME=/usr/local/cuda-12.5 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
python -m pip install --no-build-isolation torch-scatter==2.1.2
```

Verify:

```bash
python - <<'PY'
import torch_scatter

print("torch-scatter:", torch_scatter.__version__)
PY
```

Expected:

```text
torch-scatter: 2.1.2
```

---

# 13. Install NeuroVFM in editable mode

Install the current source tree:

```bash
cd /mnt/work/repo/neurovfm && \
python -m pip install -e .
```

A successful installation ends with:

```text
Successfully built neurovfm
Successfully installed neurovfm-0.1.0
```

An editable installation points the Python environment to the current source directory. It does not copy the complete NeuroVFM source tree into the environment.

Therefore, do not delete or move:

```text
/mnt/work/repo/neurovfm
```

without reinstalling NeuroVFM from its new location afterward.

Verify the active source path:

```bash
python - <<'PY'
import neurovfm

print("NeuroVFM source:", neurovfm.__file__)
PY
```

---

# 14. Verify all dependencies

This test checks PyTorch, FlashAttention, `fused_dense_lib`, `torch-scatter`, and NeuroVFM together.

```bash
python - <<'PY'
import torch
import flash_attn
import torch_scatter
import neurovfm

from flash_attn.ops.fused_dense import FusedDense
from neurovfm.pipelines.encoder import load_encoder

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("flash-attn:", flash_attn.__version__)
print("torch-scatter:", torch_scatter.__version__)
print("NeuroVFM source:", neurovfm.__file__)
print("FusedDense OK")
print("NeuroVFM import OK")
PY
```

---

# 15. Add the NeuroVFM checkpoint

The checkpoint directory must contain the files expected by `load_encoder`, including:

```text
ckpt/
├── config.json
└── pytorch_model.bin
```

If the checkpoint is stored inside the repository, verify it from the repository root:

```bash
cd /mnt/work/repo/neurovfm && \
test -f ckpt/config.json && \
test -f ckpt/pytorch_model.bin && \
echo "Checkpoint files found"
```

If `load_encoder("ckpt")` cannot find a valid local directory, the function may interpret `"ckpt"` as a Hugging Face repository ID and attempt to access:

```text
huggingface.co/ckpt
```

This results in a repository-not-found or authorization error.

To avoid ambiguity, an absolute checkpoint path may be used.

---

# 16. Load the NeuroVFM encoder

From the repository root:

```bash
cd /mnt/work/repo/neurovfm && \
python - <<'PY'
from neurovfm.pipelines.encoder import load_encoder

encoder, preproc = load_encoder("ckpt")

print("ENCODER LOADED OK")
print(type(encoder.model))
PY
```

Expected:

```text
ENCODER LOADED OK
<class 'neurovfm.models.vit.VisionTransformer'>
```

This confirms that:

```text
NeuroVFM imports correctly
The checkpoint can be read
The Vision Transformer can be constructed
FlashAttention dependencies load correctly
fused_dense_lib loads correctly
torch-scatter loads correctly
```

---

# 17. Run an end-to-end MRI embedding test

Replace the checkpoint and scan paths with real absolute paths.

```bash
cd /mnt/work/repo/neurovfm && \
python - <<'PY'
from neurovfm.pipelines.encoder import load_encoder

checkpoint = "/mnt/work/repo/neurovfm/ckpt"
scan = "/absolute/path/to/real_scan.nii.gz"

encoder, preproc = load_encoder(checkpoint)

batch = preproc.load_study(
    scan,
    modality="mri",
)

embs = encoder.embed(batch)

print("embeddings shape:", embs.shape)
print("dtype:", embs.dtype)
print("device:", embs.device)
PY
```

The exact number of output tokens depends on the input volume and background-token removal.

For the previously tested BraTS MRI, the output was:

```text
torch.Size([848, 768])
```

The feature dimension was 768.

---

# 18. Files that can be removed

The following are generated artifacts and can be regenerated:

```text
neurovfm.egg-info/
build/
dist/
__pycache__/
*.pyc
```

Files named `fd` and `inspect` found during the original setup were PostScript documents generated by ImageMagick. They were unrelated to NeuroVFM and not required by the environment.

Suggested `.gitignore` entries:

```gitignore
# Python
__pycache__/
*.py[cod]

# Packaging
*.egg-info/
build/
dist/

# Local artifacts
fd
inspect

# Large or private model files
ckpt/
```

Only ignore `ckpt/` if checkpoints should not be committed to the repository.

---

# 19. Completed-environment verification

After restarting the computer or opening a new terminal, run:

```bash
conda activate ~/conda-envs/nvfm && \
export CUDA_HOME=/usr/local/cuda-12.5 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
cd /mnt/work/repo/neurovfm && \
python - <<'PY'
import torch
import flash_attn
import torch_scatter
import neurovfm

from flash_attn.ops.fused_dense import FusedDense
from neurovfm.pipelines.encoder import load_encoder

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("flash-attn:", flash_attn.__version__)
print("torch-scatter:", torch_scatter.__version__)
print("NeuroVFM source:", neurovfm.__file__)
print("FusedDense OK")

encoder, preproc = load_encoder("ckpt")

print("ENCODER LOADED OK")
print(type(encoder.model))
PY
```

The definitive success output is:

```text
ENCODER LOADED OK
<class 'neurovfm.models.vit.VisionTransformer'>
```

---

# Troubleshooting reference

## `nvcc: command not found`

Set CUDA for the current shell:

```bash
export CUDA_HOME=/usr/local/cuda-12.5
export PATH="$CUDA_HOME/bin:$PATH"
```

Verify:

```bash
which nvcc
nvcc --version
```

---

## `fatal error: cusparse.h: No such file or directory`

The complete toolkit headers are missing. Install the full toolkit:

```bash
sudo apt install -y cuda-toolkit-12-5
```

Verify:

```bash
ls /usr/local/cuda-12.5/include/cusparse.h
```

---

## `ModuleNotFoundError: No module named 'psutil'`

Install FlashAttention's build prerequisites:

```bash
python -m pip install psutil packaging ninja wheel setuptools
```

---

## FlashAttention remains at `still running`

Check compilation activity in another terminal:

```bash
top
```

Active `ptxas`, `nvcc`, `ninja`, `c++`, or `cc1plus` processes indicate that compilation is progressing.

---

## `No module named 'fused_dense_lib'`

Build the separate extension:

```bash
cd /mnt/work/repo/flash-attention-2.6.3/csrc/fused_dense_lib && \
export CUDA_HOME=/usr/local/cuda-12.5 && \
export PATH="$CUDA_HOME/bin:$PATH" && \
python -m pip install -v --no-build-isolation .
```

---

## `No module named 'torch'` while building torch-scatter

Install `torch-scatter` separately without build isolation:

```bash
python -m pip install --no-build-isolation torch-scatter==2.1.2
```

---

## NeuroVFM rejects Python 3.10.x

Change the requirement in `pyproject.toml` from:

```toml
requires-python = "==3.10.14"
```

to:

```toml
requires-python = ">=3.10,<3.11"
```

Then retry:

```bash
python -m pip install -e .
```

---

## `load_encoder("ckpt")` contacts Hugging Face

The local checkpoint path was not found or was not recognized.

Verify:

```bash
cd /mnt/work/repo/neurovfm && \
ls -l ckpt/config.json ckpt/pytorch_model.bin
```

Alternatively, use an absolute checkpoint path:

```python
encoder, preproc = load_encoder(
    "/mnt/work/repo/neurovfm/ckpt"
)
```

---

## NeuroVFM imports from an unexpected repository

Check the active editable-install path:

```bash
python - <<'PY'
import neurovfm
print(neurovfm.__file__)
PY
```

If necessary, reinstall from the intended repository:

```bash
cd /mnt/work/repo/neurovfm && \
python -m pip install -e .
```