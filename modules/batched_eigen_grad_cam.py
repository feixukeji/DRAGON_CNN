"""Device-native, batched EigenGradCAM for CNN classifiers."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ActivationCapture:
    """Own the forward hook used to retain one target-layer activation."""

    def __init__(self, target_layer: nn.Module) -> None:
        self._activation: Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(
        self,
        _module: nn.Module,
        _inputs: tuple[object, ...],
        output: object,
    ) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("EigenGradCAM target layer must return a Tensor.")
        self._activation = output

    @property
    def activation(self) -> Tensor:
        if self._activation is None:
            raise RuntimeError("The target layer did not produce an activation.")
        return self._activation

    def clear(self) -> None:
        self._activation = None

    def close(self) -> None:
        self.clear()
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def batched_eigen_projection(weighted_activations: Tensor) -> Tensor:
    """Return the first principal-component image for a batch of feature maps."""
    if weighted_activations.ndim != 4:
        raise ValueError(
            "Expected weighted activations with shape (batch, channels, height, width), "
            f"got {tuple(weighted_activations.shape)}."
        )
    if weighted_activations.shape[0] == 0:
        raise ValueError("Cannot compute EigenGradCAM for an empty batch.")

    batch_size, _, height, width = weighted_activations.shape
    finite_activations = torch.nan_to_num(
        weighted_activations,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    matrices = finite_activations.flatten(start_dim=2).transpose(1, 2)
    if matrices.dtype in (torch.float16, torch.bfloat16):
        matrices = matrices.float()
    matrices = matrices - matrices.mean(dim=1, keepdim=True)

    _, _, vh = torch.linalg.svd(matrices, full_matrices=False)
    principal_directions = vh[:, 0, :].unsqueeze(-1)
    projections = torch.bmm(matrices, principal_directions).squeeze(-1)
    return projections.reshape(batch_size, height, width)


def _normalize_cam(cam: Tensor) -> Tensor:
    """Min-max normalize every CAM independently on its current device."""
    cam = cam - cam.amin(dim=(-2, -1), keepdim=True)
    scale = cam.amax(dim=(-2, -1), keepdim=True)
    epsilon = torch.finfo(cam.dtype).eps
    return cam / scale.clamp_min(epsilon)


class BatchedEigenGradCAM:
    """Compute EigenGradCAM without leaving PyTorch or the model device."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model.eval()
        self.device = next(self.model.parameters()).device
        self._capture = _ActivationCapture(target_layer)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Return normalized CAMs with shape ``(batch, height, width)``."""
        if input_tensor.ndim != 4:
            raise ValueError(
                f"Expected input shape (batch, channels, height, width), "
                f"got {tuple(input_tensor.shape)}."
            )

        input_tensor = input_tensor.to(self.device, non_blocking=True)
        self._capture.clear()
        try:
            with torch.enable_grad():
                outputs = self.model(input_tensor)
                if outputs.ndim != 2:
                    raise ValueError(
                        f"Expected classifier output shape (batch, classes), "
                        f"got {tuple(outputs.shape)}."
                    )

                activations = self._capture.activation
                target_categories = outputs.detach().argmax(dim=1, keepdim=True)
                target_scores = outputs.gather(1, target_categories).sum()
                gradients, = torch.autograd.grad(target_scores, activations)

            with torch.no_grad():
                weighted_activations = activations.detach() * gradients.detach()
                cam = batched_eigen_projection(weighted_activations).clamp_min_(0)
                cam = F.interpolate(
                    cam.unsqueeze(1),
                    size=input_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                return _normalize_cam(cam)
        finally:
            self._capture.clear()

    def __call__(self, input_tensor: Tensor) -> Tensor:
        return self.forward(input_tensor)

    def close(self) -> None:
        self._capture.close()

    def __enter__(self) -> "BatchedEigenGradCAM":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> bool:
        self.close()
        return False


__all__ = ["BatchedEigenGradCAM", "batched_eigen_projection"]
