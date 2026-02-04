#!/usr/bin/env python
import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.nn.utils import parametrize


def _fast_walsh_hadamard(x: Tensor) -> Tensor:
    assert x.dim() == 1
    size = x.shape[0]
    assert size == 2 ** int(round(math.log2(size)))
    h = 1
    result = x
    while h < size:
        result = result.view(-1, h * 2)
        left = result[:, :h]
        right = result[:, h : 2 * h]
        result = torch.cat((left + right, left - right), dim=1)
        h *= 2
    return result.view(size)


class _FastFoodProjection(nn.Module):
    def __init__(self, flat_weight_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        assert flat_weight_dim > 0
        self.flat_weight_dim = flat_weight_dim
        self.size = 1 << int(math.ceil(math.log2(flat_weight_dim)))
        self.register_buffer("G", torch.randn(self.size, device=device, dtype=dtype))
        self.register_buffer("Pi", torch.randperm(self.size, device=device))
        self.register_buffer("B", (torch.randint(0, 2, (self.size,), device=device) * 2 - 1).to(dtype))
        divisor = torch.sqrt(self.size * torch.sum(torch.pow(self.G, 2)))
        self.register_buffer("divisor", divisor)

    def forward(self, theta: Tensor) -> Tensor:
        assert theta.dim() == 1
        theta_padded = F.pad(theta, pad=(0, self.size - theta.size(0)), value=0.0, mode="constant")
        theta_padded = theta_padded * self.B
        hbx = _fast_walsh_hadamard(theta_padded)
        pihbx = hbx[self.Pi]
        gpihbx = pihbx * self.G
        hgpihbx = _fast_walsh_hadamard(gpihbx)
        result = hgpihbx[: self.flat_weight_dim]
        result = result / (self.divisor * math.sqrt(float(self.flat_weight_dim) / self.size))
        return result


class _SubspaceParametrization(nn.Module):
    def __init__(self, theta: nn.Parameter, param_shape: tuple[int, ...]) -> None:
        super().__init__()
        assert theta.dim() == 1
        flat_dim = int(np.prod(param_shape))
        object.__setattr__(self, "_theta", theta)
        self._param_shape = param_shape
        self._proj = _FastFoodProjection(flat_dim, theta.device, theta.dtype)

    def forward(self, weight: Tensor) -> Tensor:
        delta = self._proj(self._theta).view(self._param_shape)
        return weight + delta


def apply_intrinsic_dimension_subspace(model: nn.Module, intrinsic_dim: int) -> nn.Parameter:
    assert intrinsic_dim > 0
    params = [(name, param.shape) for name, param in model.named_parameters()]
    assert params
    first_param = next(model.parameters())
    theta = nn.Parameter(torch.zeros(intrinsic_dim, device=first_param.device, dtype=first_param.dtype))
    model.theta = theta
    for full_name, param_shape in params:
        module_name, param_name = full_name.rsplit(".", 1) if "." in full_name else ("", full_name)
        module = model.get_submodule(module_name) if module_name else model
        assert not parametrize.is_parametrized(module, param_name)
        parametrize.register_parametrization(
            module,
            param_name,
            _SubspaceParametrization(theta, param_shape),
        )
        module.parametrizations[param_name].original.requires_grad_(False)
    return theta


class ToyObjective(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim))

    def forward(self) -> Tensor:
        return self.weight


def main() -> None:
    for intrinsic_dim in range(2, 40, 2):
        torch.manual_seed(0)
        dim = 1000
        groups = 10
        group_size = dim // groups
        assert group_size * groups == dim
        steps = 10000
        lr = 0.1

        model = ToyObjective(dim)
        theta = apply_intrinsic_dimension_subspace(model, intrinsic_dim)
        optimizer = torch.optim.SGD([theta], lr=lr)

        targets = torch.arange(1, groups + 1, dtype=model.weight.dtype, device=model.weight.device)

        for _ in range(steps):
            weight = model()
            group_sums = weight.view(groups, group_size).sum(dim=1)
            loss = torch.mean((group_sums - targets) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        weight = model().detach()
        group_sums = weight.view(groups, group_size).sum(dim=1)
        loss = torch.mean((group_sums - targets) ** 2).item()
        print(f"intrinsic dim: {intrinsic_dim}")
        print(f"final loss: {loss:.6f}")
        print("group sums:", group_sums.cpu().numpy().round(3))


if __name__ == "__main__":
    main()
