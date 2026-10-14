import math
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Safely read a key from either a Python dict or a pyhocon ConfigTree."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            val = cfg.get(key, default)
            return default if val is None else val
        except Exception:
            pass
    try:
        return cfg[key]
    except (KeyError, IndexError, TypeError):
        return default


class MultiResHashGrid(nn.Module):
    """Pure-PyTorch multi-resolution hash encoding (Instant-NGP / tiny-cuda-nn replacement).

    Supports any PyTorch device (CUDA, ROCm, Apple MPS, Intel XPU, CPU) without
    custom C++/CUDA extensions. Stores each resolution level as a separate
    Parameter in a ParameterList to avoid full-table gradient allocations during
    backward passes and skips inactive levels beyond the progressive bandwidth cap.
    """

    # 32-bit signed equivalents of Mueller et al. spatial hash primes:
    # 2654435761 (0x9E3779B1) -> -1640531535 in int32
    # 805459861  (0x30021E15) ->  805459861  in int32
    PRIME_Y_I32: int = -1640531535
    PRIME_Z_I32: int = 805459861

    def __init__(
        self,
        d_in: int = 3,
        encoding_config: Optional[Union[Dict[str, Any], Any]] = None,
        bbox_min: float = -1.0,
        bbox_max: float = 1.0,
    ) -> None:
        super().__init__()
        if d_in != 3:
            raise ValueError(f"MultiResHashGrid expects 3D inputs (d_in=3), got {d_in}.")

        self.d_in = d_in
        self.n_levels = int(_cfg_get(encoding_config, "n_levels", 14))
        self.n_features_per_level = int(_cfg_get(encoding_config, "n_features_per_level", 2))
        self.log2_hashmap_size = int(_cfg_get(encoding_config, "log2_hashmap_size", 19))
        self.base_resolution = float(_cfg_get(encoding_config, "base_resolution", 32.0))
        self.per_level_scale = float(
            _cfg_get(encoding_config, "per_level_scale", 1.3195079107728942)
        )

        self.hashmap_size = 1 << self.log2_hashmap_size
        self.n_output_dims = self.n_levels * self.n_features_per_level
        self.bbox_min = float(bbox_min)
        self.bbox_max = float(bbox_max)

        resolutions = []
        level_is_dense = []
        embeddings = nn.ParameterList()

        for level in range(self.n_levels):
            res = int(math.floor(self.base_resolution * (self.per_level_scale ** level)))
            resolutions.append(res)
            verts_per_level = (res + 1) ** 3
            if verts_per_level <= self.hashmap_size:
                level_size = verts_per_level
                level_is_dense.append(True)
            else:
                level_size = self.hashmap_size
                level_is_dense.append(False)

            param = nn.Parameter(
                torch.empty(level_size, self.n_features_per_level, dtype=torch.float32)
            )
            nn.init.uniform_(param, -1e-4, 1e-4)
            embeddings.append(param)

        self.level_resolutions = resolutions
        self.level_is_dense = level_is_dense
        self.embeddings = embeddings

    def forward(self, inputs: torch.Tensor, max_active_levels: Optional[int] = None) -> torch.Tensor:
        """Trilinearly interpolate multi-resolution hash features for 3D points.

        Args:
            inputs: Tensor of shape (N, 3) in scene coordinates (typically [-1, 1]^3).
            max_active_levels: Optional progressive bandwidth cap (number of active levels).

        Returns:
            Encoded feature tensor of shape (N, n_levels * n_features_per_level).
        """
        orig_shape = inputs.shape[:-1]
        x = inputs.reshape(-1, 3).to(torch.float32)
        n_pts = x.shape[0]

        if n_pts == 0:
            return torch.zeros(
                (*orig_shape, self.n_output_dims), dtype=torch.float32, device=inputs.device
            )

        # Normalize from [bbox_min, bbox_max] -> [0, 1]
        inv_extent = 1.0 / (self.bbox_max - self.bbox_min)
        x_norm = ((x - self.bbox_min) * inv_extent).clamp(0.0, 1.0 - 1e-7)

        active_levels = (
            self.n_levels
            if max_active_levels is None
            else max(0, min(self.n_levels, int(max_active_levels)))
        )

        hash_mask = int(self.hashmap_size - 1)
        f_dim = self.n_features_per_level
        level_outputs = []

        for level in range(active_levels):
            res = self.level_resolutions[level]
            is_dense = self.level_is_dense[level]
            table = self.embeddings[level]

            # Scale normalized positions to current grid resolution
            pos = x_norm * float(res)  # (N, 3)
            pos_floor = torch.floor(pos)
            w = pos - pos_floor  # (N, 3) fractional weights in [0, 1)

            wx1, wy1, wz1 = w[:, 0:1], w[:, 1:2], w[:, 2:3]
            wx0, wy0, wz0 = 1.0 - wx1, 1.0 - wy1, 1.0 - wz1

            # 8 trilinear weights: (N, 1) each
            w000 = wx0 * wy0 * wz0
            w100 = wx1 * wy0 * wz0
            w010 = wx0 * wy1 * wz0
            w110 = wx1 * wy1 * wz0
            w001 = wx0 * wy0 * wz1
            w101 = wx1 * wy0 * wz1
            w011 = wx0 * wy1 * wz1
            w111 = wx1 * wy1 * wz1

            if is_dense:
                pos_int = pos_floor.long().clamp(0, res - 1)
                stride = res + 1
                stride2 = stride * stride
                ix0 = pos_int[:, 0]
                iy0 = pos_int[:, 1] * stride
                iz0 = pos_int[:, 2] * stride2
                ix1 = ix0 + 1
                iy1 = iy0 + stride
                iz1 = iz0 + stride2

                idx000 = ix0 + iy0 + iz0
                idx100 = ix1 + iy0 + iz0
                idx010 = ix0 + iy1 + iz0
                idx110 = ix1 + iy1 + iz0
                idx001 = ix0 + iy0 + iz1
                idx101 = ix1 + iy0 + iz1
                idx011 = ix0 + iy1 + iz1
                idx111 = ix1 + iy1 + iz1
            else:
                pos_int = pos_floor.to(torch.int32)
                ix0 = pos_int[:, 0]
                ix1 = ix0 + 1
                iy0 = pos_int[:, 1] * self.PRIME_Y_I32
                iy1 = (pos_int[:, 1] + 1) * self.PRIME_Y_I32
                iz0 = pos_int[:, 2] * self.PRIME_Z_I32
                iz1 = (pos_int[:, 2] + 1) * self.PRIME_Z_I32

                xy00 = torch.bitwise_xor(ix0, iy0)
                xy10 = torch.bitwise_xor(ix1, iy0)
                xy01 = torch.bitwise_xor(ix0, iy1)
                xy11 = torch.bitwise_xor(ix1, iy1)

                idx000 = (torch.bitwise_xor(xy00, iz0) & hash_mask).long()
                idx100 = (torch.bitwise_xor(xy10, iz0) & hash_mask).long()
                idx010 = (torch.bitwise_xor(xy01, iz0) & hash_mask).long()
                idx110 = (torch.bitwise_xor(xy11, iz0) & hash_mask).long()
                idx001 = (torch.bitwise_xor(xy00, iz1) & hash_mask).long()
                idx101 = (torch.bitwise_xor(xy10, iz1) & hash_mask).long()
                idx011 = (torch.bitwise_xor(xy01, iz1) & hash_mask).long()
                idx111 = (torch.bitwise_xor(xy11, iz1) & hash_mask).long()

            interp_feat = (
                w000 * table.index_select(0, idx000)
                + w100 * table.index_select(0, idx100)
                + w010 * table.index_select(0, idx010)
                + w110 * table.index_select(0, idx110)
                + w001 * table.index_select(0, idx001)
                + w101 * table.index_select(0, idx101)
                + w011 * table.index_select(0, idx011)
                + w111 * table.index_select(0, idx111)
            )
            level_outputs.append(interp_feat)

        if active_levels < self.n_levels:
            inactive_dims = (self.n_levels - active_levels) * f_dim
            # Keep a zero-multiplied link to embeddings[0] when active_levels == 0
            # so SummaryWriter.add_graph sees a connected graph before iter 0
            zero_anchor = self.embeddings[0][:1, :1].sum() * 0.0
            zeros_inactive = zero_anchor.expand(n_pts, inactive_dims)
            level_outputs.append(zeros_inactive)

        encoded = torch.cat(level_outputs, dim=-1)
        return encoded.reshape(*orig_shape, self.n_output_dims)


class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_hidden: int,
        n_layers: int,
        skip_in: Sequence[int] = (4,),
        bias: float = 0.5,
        geometric_init: bool = True,
        weight_norm: bool = True,
        inside_outside: bool = False,
        encoding_config: Optional[Any] = None,
        input_concat: bool = False,
    ) -> None:
        super(SDFNetwork, self).__init__()
        self.input_concat = input_concat

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        if encoding_config is not None:
            self.encoding = MultiResHashGrid(d_in, encoding_config).to(torch.float32)
            dims[0] = self.encoding.n_output_dims
            if input_concat:
                dims[0] += d_in
        else:
            self.encoding = None

        self.num_layers = len(dims)
        self.skip_in = tuple(skip_in)

        self.bindwidth = 0
        self.enc_dim = self.encoding.n_output_dims if self.encoding is not None else 0

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif self.encoding is not None and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif self.encoding is not None and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def increase_bandwidth(self) -> None:
        self.bindwidth += 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.encoding is not None:
            encoded = self.encoding(inputs, max_active_levels=self.bindwidth).to(torch.float32)

        if self.input_concat and self.encoding is not None:
            inputs = torch.cat([inputs, encoded], dim=1)
        elif self.encoding is not None:
            inputs = encoded

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return x

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    @torch.enable_grad()
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        x = x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.unsqueeze(1)

    @torch.enable_grad()
    def divergence(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        div = 0.0
        for i in range(y.shape[-1]):
            div += torch.autograd.grad(
                y[..., i], x, torch.ones_like(y[..., i]), create_graph=True
            )[0][..., i : i + 1]
        return div

    @torch.enable_grad()
    def laplace(self, x: torch.Tensor) -> torch.Tensor:
        return self.divergence(self.gradient(x), x)


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val: float) -> None:
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones([len(x), 1], device=self.variance.device, dtype=self.variance.dtype) * torch.exp(
            self.variance * 10.0
        )