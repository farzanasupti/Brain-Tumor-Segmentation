"""Swin-UMamba, the Mamba arm of the factorial.

This arm is the reason the study exists: the augmentation under test was
introduced on a Mamba backbone, and without it there is no way to tell whether
the effect is backbone-agnostic or specific to that family.

It is also the hardest to install. The model lives in the Swin-UMamba
repository rather than on PyPI, and depends on mamba_ssm, which compiles CUDA
kernels against the installed PyTorch. Both failure modes are reported by
check() with the fix rather than left to surface mid-run.

Setup:

    git clone https://github.com/JiarunLiu/Swin-UMamba.git
    pip install -e Swin-UMamba/swin_umamba
    pip install packaging ninja
    pip install mamba_ssm

Set BTS_SWIN_UMAMBA to the repository path if it is not importable already.

On mamba_ssm build failures, in order of likelihood:

  1. No GPU runtime attached. mamba_ssm's setup.py reads torch.version.cuda and
     crashes when it is None -- on a CPU-only PyTorch it fails during metadata
     generation with "TypeError: expected string or bytes-like object, got
     'NoneType'", which looks nothing like a missing-GPU message. Attach a GPU
     runtime first and confirm torch.version.cuda is not None.
  2. Missing build tools: pip install packaging ninja, before mamba_ssm.
  3. Source build times out or exhausts memory. Prefer a prebuilt wheel from
     the mamba_ssm GitHub releases matching your Python, torch and CUDA
     versions over compiling from source.
"""

import importlib
import os
import sys

ENV_PATH = "BTS_SWIN_UMAMBA"


def _add_repo_to_path():
    """Put a locally cloned Swin-UMamba on sys.path if one is configured."""
    root = os.environ.get(ENV_PATH)
    if not root:
        return None
    for candidate in (os.path.join(root, "swin_umamba"), root):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)
    return root


def _import_model():
    _add_repo_to_path()
    module = importlib.import_module("nnunetv2.nets.SwinUMamba")
    return getattr(module, "SwinUMamba")


def check():
    try:
        import torch
    except ImportError:
        return False, "torch not installed"

    cuda = getattr(torch.version, "cuda", None)

    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        if cuda is None:
            return False, (
                "mamba_ssm missing and this PyTorch has no CUDA "
                f"(torch {torch.__version__}, cuda=None). Its setup.py parses "
                "torch.version.cuda and fails obscurely when it is None -- "
                "attach a GPU runtime before installing."
            )
        return False, (
            f"mamba_ssm not installed (torch {torch.__version__}, cuda {cuda}). "
            "Run: pip install packaging ninja && pip install mamba_ssm, or fetch "
            "a prebuilt wheel matching this Python/torch/CUDA."
        )
    except Exception as exc:                        # noqa: BLE001
        return False, f"mamba_ssm present but unusable: {type(exc).__name__}: {exc}"

    try:
        _import_model()
    except ImportError as exc:
        hint = (f"set {ENV_PATH} to the clone" if not os.environ.get(ENV_PATH)
                else f"{ENV_PATH}={os.environ[ENV_PATH]} did not expose it")
        return False, (
            f"mamba_ssm is installed but nnunetv2.nets.SwinUMamba is not "
            f"importable ({exc}); clone JiarunLiu/Swin-UMamba and {hint}."
        )
    except Exception as exc:                        # noqa: BLE001
        return False, f"SwinUMamba import failed: {type(exc).__name__}: {exc}"

    return True, f"torch {torch.__version__}, cuda {cuda}"


def build(in_channels=3, out_channels=1, img_size=256, **kwargs):
    ok, detail = check()
    if not ok:
        raise ImportError(
            f"Swin-UMamba is unavailable: {detail}\n"
            f"See the setup notes at the top of {__file__}."
        )

    SwinUMamba = _import_model()
    return SwinUMamba(in_chans=in_channels, num_classes=out_channels, **kwargs)
