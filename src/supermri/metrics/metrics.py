import torch
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F


def mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))


def mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def nmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Normalized mean squared error
    Note: normalize by y_true, not y_pred.
    """
    return mse(y_pred, y_true) / (y_true.norm() ** 2 + 1e-6)


def psnr(x: torch.Tensor, y: torch.Tensor, max_value: float = 1.0) -> torch.Tensor:
    """Computes peak signal to noise ratio (PSNR)
    Args:
      x: images in (N,C,H,W)
      y: images in (N,C,H,W)
      max_value: the maximum value of the images (usually 1.0 or 255.0)
    Returns:
      PSNR value
    """
    return 10 * torch.log10(max_value**2 / (mse(x, y) + 1e-6))


def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    """Create 1-D Gaussian kernel with shape (1, 1, size)
    Args:
      size: the size of the Gaussian kernel
      sigma: sigma of normal distribution
    Returns:
      1D kernel (1, 1, size)
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def _gaussian_filter(inputs: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Apply 1D Gaussian kernel to inputs images
    Args:
      inputs: a batch of images in shape (N,C,H,W)
      win: 1-D Gaussian kernel
    Returns:
      blurred images
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    channel = inputs.shape[1]
    outputs = inputs
    for i, s in enumerate(inputs.shape[2:]):
        if s >= win.shape[-1]:
            outputs = F.conv2d(
                outputs,
                weight=win.transpose(2 + i, -1),
                stride=1,
                padding=0,
                groups=channel,
            )
    return outputs


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    max_value: float = 1.0,
    size_average: bool = True,
    win_size: int = 11,
    win_sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
) -> torch.Tensor:
    """Computes structural similarity index metric (SSIM)

    Reference: https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py

    Args:
      x: images in the format of (N,C,H,W)
      y: images in the format of (N,C,H,W)
      max_value: the maximum value of the images (usually 1.0 or 255.0)
      size_average: return SSIM average of all images
      win_size: the size of gauss kernel
      win_sigma: sigma of normal distribution
      win: 1-D gauss kernel. if None, a new kernel will be created according to
          win_size and win_sigma
      K1: scalar constant
      K2: scalar constant
    Returns:
      SSIM value(s)
    """
    assert x.shape == y.shape, "input images should have the same dimensions."

    # remove dimensions that has size 1, except the batch and channel dimensions
    for d in range(2, x.ndim):
        x = x.squeeze(dim=d)
        y = y.squeeze(dim=d)

    assert x.ndim == 4, f"input images should be 4D, but got {x.ndim}."
    assert win_size % 2 == 1, f"win_size should be odd, but got {win_size}."

    win = _gaussian_kernel_1d(win_size, win_sigma)
    win = win.repeat([x.shape[1]] + [1] * (len(x.shape) - 1))

    compensation = 1.0

    C1 = (K1 * max_value) ** 2
    C2 = (K2 * max_value) ** 2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = _gaussian_filter(x, win)
    mu2 = _gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (_gaussian_filter(x * x, win) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(y * y, win) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter(x * y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)

    return ssim_per_channel.mean() if size_average else ssim_per_channel.mean(1)
