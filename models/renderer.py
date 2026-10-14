from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mcubes
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm


def get_default_device() -> torch.device:
    """Detect the best available accelerator device across vendors."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class ContractionType(Enum):
    """Scene coordinate contraction modes matching NerfAcc."""

    AABB = 0
    UN_BOUNDED_TANH = 1
    UN_BOUNDED_SPHERE = 2


def _contract_inv(
    x: Tensor,
    roi: Tensor,
    contraction_type: ContractionType = ContractionType.AABB,
) -> Tensor:
    """Map normalized grid coordinates in [0, 1]^3 back to world coordinates."""
    roi_min = roi[:3]
    roi_max = roi[3:]
    if contraction_type == ContractionType.AABB:
        return x * (roi_max - roi_min) + roi_min
    if contraction_type == ContractionType.UN_BOUNDED_TANH:
        return torch.atanh((x * 2.0 - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)) * (
            0.5 * (roi_max - roi_min)
        ) + 0.5 * (roi_min + roi_max)
    if contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        z = x * 2.0 - 1.0
        norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        mask = (norm > 0.5).squeeze(-1)
        z_out = z * 2.0
        z_out[mask] = z[mask] / (norm[mask] * (2.0 - 2.0 * norm[mask]).clamp(min=1e-6))
        return z_out * (0.5 * (roi_max - roi_min)) + 0.5 * (roi_min + roi_max)
    raise ValueError(f"Unsupported ContractionType: {contraction_type}")


def _meshgrid3d(res: Tensor, device: Union[torch.device, str] = "cpu") -> Tensor:
    res_list = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res_list[0], dtype=torch.long, device=device),
                torch.arange(res_list[1], dtype=torch.long, device=device),
                torch.arange(res_list[2], dtype=torch.long, device=device),
            ],
            indexing="ij",
        ),
        dim=-1,
    )


class OccupancyGrid(nn.Module):
    """Pure-PyTorch 3D occupancy grid replacing nerfacc.OccupancyGrid."""

    NUM_DIM: int = 3

    def __init__(
        self,
        roi_aabb: Union[List[float], Tensor],
        resolution: Union[int, List[int], Tensor] = 128,
        contraction_type: ContractionType = ContractionType.AABB,
    ) -> None:
        super().__init__()
        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)

        self.num_cells = int(resolution.prod().item())
        self._contraction_type = contraction_type

        self.register_buffer("_dummy", torch.empty(0), persistent=False)
        self.register_buffer("_roi_aabb", roi_aabb.to(torch.float32))
        self.register_buffer("_binary", torch.zeros(resolution.tolist(), dtype=torch.bool))
        self.register_buffer("resolution", resolution.to(torch.int32))
        self.register_buffer("occs", torch.zeros(self.num_cells, dtype=torch.float32))

        grid_coords = _meshgrid3d(resolution).reshape(self.num_cells, self.NUM_DIM)
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        grid_indices = torch.arange(self.num_cells, dtype=torch.long)
        self.register_buffer("grid_indices", grid_indices, persistent=False)

    @property
    def device(self) -> torch.device:
        return self._roi_aabb.device

    @property
    def roi_aabb(self) -> Tensor:
        return self._roi_aabb

    @property
    def binary(self) -> Tensor:
        return self._binary

    @property
    def contraction_type(self) -> ContractionType:
        return self._contraction_type

    @torch.no_grad()
    def _get_all_cells(self) -> Tensor:
        return self.grid_indices

    @torch.no_grad()
    def _sample_uniform_and_occupied_cells(self, n: int) -> Tensor:
        uniform_indices = torch.randint(self.num_cells, (n,), device=self.device)
        occupied_indices = torch.nonzero(self._binary.flatten(), as_tuple=False)[:, 0]
        if n < len(occupied_indices):
            selector = torch.randint(len(occupied_indices), (n,), device=self.device)
            occupied_indices = occupied_indices[selector]
        return torch.cat([uniform_indices, occupied_indices], dim=0)

    @torch.no_grad()
    def _update(
        self,
        step: int,
        occ_eval_fn: Callable[[Tensor], Tensor],
        occ_thre: float = 0.01,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
    ) -> None:
        if step < warmup_steps:
            indices = self._get_all_cells()
        else:
            n_sample = self.num_cells // 4
            indices = self._sample_uniform_and_occupied_cells(n_sample)

        # Evaluate occupancy in VRAM-safe mini-batches (262,144 points per batch)
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords.to(torch.float32)
            + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.resolution.to(torch.float32)

        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]

        x = _contract_inv(x, roi=self._roi_aabb, contraction_type=self._contraction_type)

        occ_chunks = []
        for x_batch in torch.split(x, 262144, dim=0):
            occ_chunks.append(occ_eval_fn(x_batch).squeeze(-1))
        occ = torch.cat(occ_chunks, dim=0)

        self.occs[indices] = torch.maximum(self.occs[indices] * ema_decay, occ)
        thre = torch.clamp(self.occs.mean(), max=occ_thre)
        self._binary = (self.occs > thre).view(self._binary.shape)

    @torch.no_grad()
    def every_n_step(
        self,
        step: int,
        occ_eval_fn: Callable[[Tensor], Tensor],
        occ_thre: float = 1e-2,
        ema_decay: float = 0.95,
        warmup_steps: int = 256,
        n: int = 16,
    ) -> None:
        if not self.training:
            raise RuntimeError("every_n_step should only be called during training.")
        if step % n == 0:
            self._update(
                step=step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=occ_thre,
                ema_decay=ema_decay,
                warmup_steps=warmup_steps,
            )

    @torch.no_grad()
    def query_occ(self, samples: Tensor) -> Tensor:
        """Query binary occupancy for 3D world points (N, 3)."""
        roi_min = self._roi_aabb[:3]
        roi_max = self._roi_aabb[3:]
        res = self.resolution.to(torch.long)
        in_aabb = (samples >= roi_min).all(dim=-1) & (samples <= roi_max).all(dim=-1)
        norm_coords = (samples - roi_min) / (roi_max - roi_min)
        idx = torch.floor(norm_coords * res.to(samples.dtype)).long()
        idx_clamped = torch.minimum(
            torch.maximum(idx, torch.zeros_like(res)), res - 1
        )
        occ = self._binary[idx_clamped[:, 0], idx_clamped[:, 1], idx_clamped[:, 2]]
        return occ & in_aabb


def ray_aabb_intersect(rays_o: Tensor, rays_d: Tensor, scene_aabb: Tensor) -> Tuple[Tensor, Tensor]:
    """Vectorized slab method for ray-AABB intersection matching intersection.cu."""
    aabb_min = scene_aabb[:3]
    aabb_max = scene_aabb[3:]
    inv_d = 1.0 / torch.where(
        rays_d.abs() < 1e-10, torch.full_like(rays_d, 1e-10), rays_d
    )
    t0 = (aabb_min - rays_o) * inv_d
    t1 = (aabb_max - rays_o) * inv_d
    t_near = torch.minimum(t0, t1).amax(dim=-1)
    t_far = torch.maximum(t0, t1).amin(dim=-1)
    hit = t_far >= t_near
    t_min = torch.where(hit, torch.clamp(t_near, min=0.0), torch.full_like(t_near, 1e10))
    t_max = torch.where(hit, t_far, torch.full_like(t_far, 1e10))
    return t_min, t_max


def _segmented_transmittance_and_weights(
    alphas: Tensor, ray_indices: Tensor
) -> Tuple[Tensor, Tensor]:
    """Compute per-ray transmittance T_i = prod_{j < i}(1 - alpha_j) and weights w_i = T_i * alpha_i.

    Works for both pixel-based alphas of shape (n_samples, 1) and patch-based
    alphas of shape (n_samples, P, 1) where samples belonging to the same ray/patch
    are contiguous in `ray_indices`. Fully differentiable on any PyTorch device.
    """
    orig_shape = alphas.shape
    if alphas.numel() == 0 or ray_indices.numel() == 0:
        return torch.ones_like(alphas), torch.zeros_like(alphas)

    if alphas.dim() == 2:
        alphas_3d = alphas.unsqueeze(1)
    elif alphas.dim() == 3:
        alphas_3d = alphas
    else:
        raise ValueError(f"Expected alphas to have 2 or 3 dims, got {alphas.shape}.")

    n_samples, patch_elems, _ = alphas_3d.shape
    device = alphas.device
    dtype = alphas.dtype

    _, counts = torch.unique_consecutive(ray_indices, return_counts=True)
    n_active_rays = counts.shape[0]
    max_steps = int(counts.max().item())

    row_starts = torch.cumsum(counts, dim=0) - counts
    group_ids = torch.repeat_interleave(
        torch.arange(n_active_rays, device=device, dtype=torch.long), counts
    )
    step_ids = (
        torch.arange(n_samples, device=device, dtype=torch.long) - row_starts[group_ids]
    )

    alphas_pad = torch.zeros(
        (n_active_rays, max_steps, patch_elems, 1), device=device, dtype=dtype
    )
    alphas_pad[group_ids, step_ids] = alphas_3d

    # Clamp (1 - alpha) to 1e-10 matching render_weight.cu line 335: fmaxf(1.f - alpha, 1e-10f)
    one_minus_alpha = (1.0 - alphas_pad).clamp(min=1e-10)
    if max_steps > 1:
        prefix_prod = torch.cumprod(one_minus_alpha[:, :-1], dim=1)
        ones_head = torch.ones(
            (n_active_rays, 1, patch_elems, 1), device=device, dtype=dtype
        )
        trans_pad = torch.cat([ones_head, prefix_prod], dim=1)
    else:
        trans_pad = torch.ones(
            (n_active_rays, 1, patch_elems, 1), device=device, dtype=dtype
        )

    weights_pad = trans_pad * alphas_pad

    transmittance = trans_pad[group_ids, step_ids].reshape(orig_shape)
    weights = weights_pad[group_ids, step_ids].reshape(orig_shape)
    return transmittance, weights


def render_weight_from_alpha(
    alphas: Tensor,
    *,
    packed_info: Optional[Tensor] = None,
    ray_indices: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Pure-PyTorch replacement for nerfacc.render_weight_from_alpha."""
    if ray_indices is None:
        raise ValueError("ray_indices must be provided.")
    _, weights = _segmented_transmittance_and_weights(alphas, ray_indices)
    return weights


def render_weight_from_alpha_patch_based(
    alphas: Tensor,
    ray_indices: Tensor,
    *,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Pure-PyTorch replacement for custom nerfacc.render_weight_from_alpha_patch_based."""
    _, weights = _segmented_transmittance_and_weights(alphas, ray_indices)
    return weights


def accumulate_along_rays(
    weights: Tensor,
    ray_indices: Tensor,
    values: Optional[Tensor] = None,
    n_rays: Optional[int] = None,
) -> Tensor:
    """Pure-PyTorch device-agnostic replacement for nerfacc.accumulate_along_rays."""
    assert ray_indices.dim() == 1 and weights.dim() == 2
    if values is not None:
        assert values.dim() == 2 and values.shape[0] == weights.shape[0]
        src = weights * values
    else:
        src = weights

    if ray_indices.numel() == 0:
        assert n_rays is not None
        return torch.zeros((n_rays, src.shape[-1]), device=weights.device, dtype=src.dtype)

    if n_rays is None:
        n_rays = int(ray_indices.max().item()) + 1

    index = ray_indices.long()[:, None].expand(-1, src.shape[-1])
    outputs = torch.zeros((n_rays, src.shape[-1]), device=src.device, dtype=src.dtype)
    outputs.scatter_add_(0, index, src)
    return outputs


def accumulate_along_rays_patch_based(
    weights: Tensor,
    ray_indices: Tensor,
    values: Optional[Tensor] = None,
    n_patches: Optional[int] = None,
) -> Tensor:
    """Pure-PyTorch device-agnostic replacement for nerfacc.accumulate_along_rays_patch_based."""
    assert ray_indices.dim() == 1 and weights.dim() == 3
    if values is not None:
        assert values.dim() == 3 and values.shape[0] == weights.shape[0]
        src = weights * values
    else:
        src = weights

    if ray_indices.numel() == 0:
        assert n_patches is not None
        return torch.zeros(
            (n_patches, src.shape[1], src.shape[-1]),
            device=weights.device,
            dtype=src.dtype,
        )

    if n_patches is None:
        n_patches = int(ray_indices.max().item()) + 1

    index = ray_indices.long()[:, None, None].expand(-1, src.shape[1], src.shape[-1])
    outputs = torch.zeros(
        (n_patches, src.shape[1], src.shape[-1]), device=src.device, dtype=src.dtype
    )
    outputs.scatter_add_(0, index, src)
    return outputs


@torch.no_grad()
def ray_marching(
    rays_o: Tensor,
    rays_d: Tensor,
    t_min: Optional[Tensor] = None,
    t_max: Optional[Tensor] = None,
    scene_aabb: Optional[Tensor] = None,
    grid: Optional[OccupancyGrid] = None,
    sigma_fn: Optional[Callable] = None,
    alpha_fn: Optional[Callable] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    stratified: bool = False,
    cone_angle: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pure-PyTorch occupancy-grid ray marching with NaN-safe sphere miss handling."""
    device = rays_o.device
    dtype = rays_o.dtype
    n_rays = rays_o.shape[0]
    step_size = float(render_step_size)

    if t_min is None or t_max is None:
        if scene_aabb is not None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, scene_aabb)
        elif grid is not None:
            t_min, t_max = ray_aabb_intersect(rays_o, rays_d, grid.roi_aabb)
        else:
            t_min = torch.zeros((n_rays,), device=device, dtype=dtype)
            t_max = torch.full((n_rays,), 1e10, device=device, dtype=dtype)
    else:
        t_min = t_min.reshape(-1).to(device=device, dtype=dtype)
        t_max = t_max.reshape(-1).to(device=device, dtype=dtype)

    # Rays that miss the unit sphere in near_far_from_sphere() have NaN t_min/t_max.
    # In ray_marching.cu, `while (t_mid < far)` evaluates to false for NaN; sanitize here.
    valid_ray = torch.isfinite(t_min) & torch.isfinite(t_max) & (t_max > t_min) & (t_max < 1e9)
    t_min = torch.where(valid_ray, t_min, torch.zeros_like(t_min))
    t_max = torch.where(valid_ray, t_max, torch.zeros_like(t_max))

    if near_plane is not None:
        t_min = torch.clamp(t_min, min=float(near_plane))
    if far_plane is not None:
        t_max = torch.clamp(t_max, max=float(far_plane))

    if stratified:
        t_min = t_min + torch.rand_like(t_min) * step_size

    valid_ray = valid_ray & (t_max > t_min)
    ray_span = torch.where(valid_ray, t_max - t_min, torch.zeros_like(t_min))
    max_span = float(ray_span.max().item()) if n_rays > 0 else 0.0
    max_steps = int(np.ceil(max_span / step_size)) if max_span > 0.0 else 0

    if max_steps <= 0:
        empty_idx = torch.zeros((0,), device=device, dtype=torch.long)
        empty_t = torch.zeros((0, 1), device=device, dtype=dtype)
        return empty_idx, empty_t, empty_t.clone()

    # March rays in memory-safe chunks so peak VRAM stays tiny even at step_size=5e-4
    ray_chunk_size = max(1, min(1024, (262144 // max(1, max_steps))))
    ray_indices_list = []
    t_starts_list = []
    t_ends_list = []

    for r_start in range(0, n_rays, ray_chunk_size):
        r_end = min(r_start + ray_chunk_size, n_rays)
        v_c = valid_ray[r_start:r_end]
        if not v_c.any():
            continue

        ro_c = rays_o[r_start:r_end]
        rd_c = rays_d[r_start:r_end]
        tmin_c = t_min[r_start:r_end]
        tmax_c = t_max[r_start:r_end]
        span_c = ray_span[r_start:r_end]

        chunk_max_span = float(span_c.max().item())
        chunk_max_steps = int(np.ceil(chunk_max_span / step_size)) if chunk_max_span > 0.0 else 0
        if chunk_max_steps <= 0:
            continue

        steps = torch.arange(chunk_max_steps, device=device, dtype=dtype).unsqueeze(0)  # (1, S)
        ts = tmin_c.unsqueeze(1) + steps * step_size  # (B, S)
        te = ts + step_size  # (B, S)
        tmid = 0.5 * (ts + te)  # (B, S)

        valid = v_c.unsqueeze(1) & (tmid < tmax_c.unsqueeze(1))

        if grid is not None:
            pts = ro_c.unsqueeze(1) + rd_c.unsqueeze(1) * tmid.unsqueeze(-1)  # (B, S, 3)
            occ = grid.query_occ(pts.reshape(-1, 3)).reshape(valid.shape)
            valid = valid & occ

        local_ray_idx, _ = torch.nonzero(valid, as_tuple=True)
        if local_ray_idx.numel() > 0:
            ray_indices_list.append(local_ray_idx + r_start)
            t_starts_list.append(ts[valid].unsqueeze(-1))
            t_ends_list.append(te[valid].unsqueeze(-1))

    if len(ray_indices_list) == 0:
        empty_idx = torch.zeros((0,), device=device, dtype=torch.long)
        empty_t = torch.zeros((0, 1), device=device, dtype=dtype)
        return empty_idx, empty_t, empty_t.clone()

    ray_indices = torch.cat(ray_indices_list, dim=0)
    t_starts = torch.cat(t_starts_list, dim=0)
    t_ends = torch.cat(t_ends_list, dim=0)

    # Filter out occluded or transparent samples using alpha_fn or sigma_fn
    if sigma_fn is not None or alpha_fn is not None:
        if sigma_fn is not None:
            sigmas = sigma_fn(t_starts, t_ends, ray_indices)
            alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
        else:
            alphas = alpha_fn(t_starts, t_ends, ray_indices)

        transmittance, _ = _segmented_transmittance_and_weights(alphas, ray_indices)
        vis_mask = transmittance >= early_stop_eps
        if alpha_thre > 0.0:
            vis_mask = vis_mask & (alphas >= alpha_thre)
        vis_mask = vis_mask.squeeze(-1)

        ray_indices = ray_indices[vis_mask]
        t_starts = t_starts[vis_mask]
        t_ends = t_ends[vis_mask]

    return ray_indices, t_starts, t_ends


def extract_geometry(
    bound_min: Tensor,
    bound_max: Tensor,
    resolution: int,
    threshold: float,
    query_func: Callable[[Tensor], Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract high-resolution meshes using spatial chunking to prevent RAM/VRAM overflow."""
    print(f"Extracting geometry at {resolution}^3 resolution using Memory-Safe Chunking...")
    if device is None:
        device = get_default_device()

    b_min_np = bound_min.detach().cpu().numpy()
    b_max_np = bound_max.detach().cpu().numpy()

    # 256^3 = 16.7 million voxels (~67 MB of RAM per chunk)
    chunk_res = 256
    # Step overlaps by 1 voxel to ensure seamless meshing across chunk boundaries
    step = chunk_res - 1

    X_global = torch.linspace(float(bound_min[0]), float(bound_max[0]), resolution)
    Y_global = torch.linspace(float(bound_min[1]), float(bound_max[1]), resolution)
    Z_global = torch.linspace(float(bound_min[2]), float(bound_max[2]), resolution)

    all_vertices = []
    all_triangles = []
    vertex_offset = 0

    x_steps = list(range(0, resolution - 1, step))
    y_steps = list(range(0, resolution - 1, step))
    z_steps = list(range(0, resolution - 1, step))
    total_chunks = len(x_steps) * len(y_steps) * len(z_steps)

    with tqdm(total=total_chunks, desc="Processing spatial chunks") as pbar:
        for xi in x_steps:
            xi_end = min(xi + chunk_res, resolution)
            xs = X_global[xi:xi_end]

            for yi in y_steps:
                yi_end = min(yi + chunk_res, resolution)
                ys = Y_global[yi:yi_end]

                for zi in z_steps:
                    zi_end = min(zi + chunk_res, resolution)
                    zs = Z_global[zi:zi_end]

                    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1
                    )

                    # Process points in VRAM-safe mini-batches (64^3 = 262,144 points per batch)
                    pts_split = torch.split(pts, 262144, dim=0)

                    val_list = []
                    with torch.no_grad():
                        for pts_batch in pts_split:
                            val = (
                                query_func(pts_batch.to(device))
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            val_list.append(val)

                    u_chunk = np.concatenate(val_list, axis=0).reshape(
                        len(xs), len(ys), len(zs)
                    )

                    # Empty space skipping: bypass Marching Cubes if chunk is completely solid or empty
                    if not (np.all(u_chunk > threshold) or np.all(u_chunk < threshold)):
                        vertices, triangles = mcubes.marching_cubes(u_chunk, threshold)

                        if len(vertices) > 0:
                            vertices[:, 0] += xi
                            vertices[:, 1] += yi
                            vertices[:, 2] += zi

                            vertices = (
                                vertices
                                / (resolution - 1.0)
                                * (b_max_np - b_min_np)[None, :]
                                + b_min_np[None, :]
                            )

                            all_vertices.append(vertices)
                            all_triangles.append(triangles + vertex_offset)
                            vertex_offset += len(vertices)

                    pbar.update(1)

    if len(all_vertices) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    print("Stitching chunks together...")
    final_vertices = np.vstack(all_vertices)
    final_triangles = np.vstack(all_triangles)
    return final_vertices, final_triangles


class NeuSRenderer:
    def __init__(
        self,
        sdf_network: nn.Module,
        deviation_network: nn.Module,
        gradient_method: str = "dfd",
    ) -> None:
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network

        try:
            self.device = next(self.sdf_network.parameters()).device
        except StopIteration:
            self.device = get_default_device()

        self.scene_aabb = torch.tensor(
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.device
        )
        self.contraction_type = ContractionType.AABB
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.scene_aabb,
            resolution=128,
            contraction_type=self.contraction_type,
        ).to(self.device)
        self.sampling_step_size = np.float64(0.01)
        self.gradient_method = gradient_method

    def occ_eval_fn(self, x: Tensor) -> Tensor:
        sdf = self.sdf_network(x)[..., :1]
        alpha = torch.sigmoid(-sdf * 80.0)
        return alpha

    def _eval_sdf_chunked(
        self, pts_flat: Tensor, chunk_size: int = 262144, use_checkpoint: bool = False
    ) -> Tensor:
        """Evaluate SDFNetwork in memory-safe mini-batches to cap peak VRAM."""
        if pts_flat.shape[0] <= chunk_size:
            if use_checkpoint:
                return torch_checkpoint(self.sdf_network, pts_flat, use_reentrant=False)
            return self.sdf_network(pts_flat)

        outs = []
        for chunk in torch.split(pts_flat, chunk_size, dim=0):
            if use_checkpoint:
                outs.append(
                    torch_checkpoint(self.sdf_network, chunk, use_reentrant=False)
                )
            else:
                outs.append(self.sdf_network(chunk))
        return torch.cat(outs, dim=0)

    def render(
        self,
        rays_o_patch_all: Tensor,
        rays_d_patch_all: Tensor,
        marching_plane_normal: Tensor,
        near: Tensor,
        far: Tensor,
        V_inverse_patch_all: Tensor,
        val_gradient_method: str = "dfd",
        mode: str = "train",
    ) -> Dict[str, Any]:
        device = rays_o_patch_all.device
        patch_H = rays_o_patch_all.shape[1]
        patch_W = rays_o_patch_all.shape[2]
        num_patch = rays_o_patch_all.shape[0]

        rays_o_patch_center = rays_o_patch_all[:, patch_H // 2, patch_W // 2]
        rays_d_patch_center = rays_d_patch_all[:, patch_H // 2, patch_W // 2]

        def alpha_fn_patch_center(
            t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor, ret_sdf: bool = False
        ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
            ray_indices = ray_indices.long()
            t_origins = rays_o_patch_center[ray_indices]
            t_dirs = rays_d_patch_center[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = torch.cat([t_starts[1:], t_starts[-1:]], dim=0)
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze(-1)
            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            positions_all = torch.cat([positions_starts, positions_ends_diff], dim=0)

            sdf_all = self._eval_sdf_chunked(positions_all, use_checkpoint=False)
            sdf_start = sdf_all[: positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0] :]

            sdf_start_shift_left = torch.cat([sdf_start[1:], sdf_start[-1:]], dim=0).clone()
            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(
                torch.zeros([1, 3], device=device)
            )[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            return alpha

        with torch.no_grad():
            patch_indices, t_starts_patch_center, t_ends_patch_center = ray_marching(
                rays_o_patch_center,
                rays_d_patch_center,
                t_min=near,
                t_max=far,
                grid=self.occupancy_grid,
                render_step_size=float(self.sampling_step_size),
                stratified=True,
                cone_angle=0.0,
                early_stop_eps=1e-8,
                alpha_fn=alpha_fn_patch_center,
            )
            samples_per_ray = patch_indices.shape[0] / num_patch
            if patch_indices.shape[0] == 0:
                return {
                    "comp_normal": torch.zeros(
                        [num_patch, patch_H, patch_W, 3], device=device
                    ),
                    "gradients": None,
                    "weight_sum": torch.zeros(
                        [num_patch, patch_H, patch_W, 1], device=device
                    ),
                    "samples_per_ray": 0.0,
                }

            num_samples = patch_indices.shape[0]
            patch_indices = patch_indices.long()

            # Compute plane-intersected sampling distances on all rays in each patch
            center_dot_m = (rays_d_patch_center * marching_plane_normal).sum(
                -1, keepdim=True
            )[patch_indices][:, None, None, :]
            all_dot_m = (
                rays_d_patch_all * marching_plane_normal[:, None, None, :]
            ).sum(-1, keepdim=True)[patch_indices]

            t_starts_patch_all = t_starts_patch_center[:, None, None, :] * center_dot_m / all_dot_m
            t_ends_patch_all = t_ends_patch_center[:, None, None, :] * center_dot_m / all_dot_m

            t_starts_patch_center_shift_left = torch.cat(
                [t_starts_patch_center[1:], t_starts_patch_center[-1:]], dim=0
            )
            diff_mask = ((t_ends_patch_center - t_starts_patch_center_shift_left) != 0)[..., 0]
            positions_starts_patch_all = (
                rays_o_patch_all[patch_indices]
                + rays_d_patch_all[patch_indices] * t_starts_patch_all
            )
            positions_ends_patch_all = (
                rays_o_patch_all[patch_indices]
                + rays_d_patch_all[patch_indices] * t_ends_patch_all
            )
            positions_ends_diff = positions_ends_patch_all[diff_mask]
            del positions_ends_patch_all
            positions_all = torch.cat([positions_starts_patch_all, positions_ends_diff], dim=0)
            del positions_ends_diff
            positions_all_shape = positions_all.shape[:-1]
            positions_all_flat = positions_all.reshape(-1, 3)
            del positions_all

        # Chunked gradient checkpointing in train mode to cap peak VRAM
        use_ckpt = (mode == "train") and torch.is_grad_enabled()
        sdf_all = self._eval_sdf_chunked(
            positions_all_flat, chunk_size=262144, use_checkpoint=use_ckpt
        )
        sdf_all = sdf_all.reshape(*positions_all_shape, 1)
        del positions_all_flat

        sdf_starts_patch_all = sdf_all[: positions_starts_patch_all.shape[0]]
        sdf_end_diff = sdf_all[positions_starts_patch_all.shape[0] :]
        sdf_ends_patch_all = torch.cat(
            [sdf_starts_patch_all[1:], sdf_starts_patch_all[-1:]], dim=0
        ).clone()
        sdf_ends_patch_all[diff_mask] = sdf_end_diff

        inv_s = self.deviation_network(
            torch.zeros([1, 3], device=device)
        )[:, :1].clip(1e-6, 1e6)

        prev_cdf = torch.sigmoid(sdf_starts_patch_all * inv_s)
        next_cdf = torch.sigmoid(sdf_ends_patch_all * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        del prev_cdf, next_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        del p, c
        weights = render_weight_from_alpha_patch_based(
            alpha.reshape(num_samples, patch_H * patch_W, 1), patch_indices
        )
        del alpha

        gradient_method = self.gradient_method if mode == "train" else val_gradient_method

        if gradient_method == "dfd":
            with torch.no_grad():
                dist_x = torch.norm(
                    positions_starts_patch_all[:, :, 1:, :]
                    - positions_starts_patch_all[:, :, :-1, :],
                    dim=-1,
                    keepdim=True,
                )
                dist_y = torch.norm(
                    positions_starts_patch_all[:, 1:, :, :]
                    - positions_starts_patch_all[:, :-1, :, :],
                    dim=-1,
                    keepdim=True,
                )

            df_dt = (sdf_ends_patch_all - sdf_starts_patch_all) / (
                t_ends_patch_all - t_starts_patch_all
            )
            df_dx = (sdf_starts_patch_all[:, :, 2:] - sdf_starts_patch_all[:, :, :-2]) / (
                dist_x[:, :, :-1] + dist_x[:, :, 1:]
            )
            df_dy = (sdf_starts_patch_all[:, 2:, :] - sdf_starts_patch_all[:, :-2, :]) / (
                dist_y[:, 1:, :] + dist_y[:, :-1, :]
            )

            df_dx_left_boundary = (
                sdf_starts_patch_all[:, :, 1:2] - sdf_starts_patch_all[:, :, 0:1]
            ) / dist_x[:, :, 0:1]
            df_dx_right_boundary = (
                sdf_starts_patch_all[:, :, -1:] - sdf_starts_patch_all[:, :, -2:-1]
            ) / dist_x[:, :, -1:]
            df_dy_top_boundary = (
                sdf_starts_patch_all[:, 1:2, :] - sdf_starts_patch_all[:, 0:1, :]
            ) / dist_y[:, 0:1, :]
            df_dy_bottom_boundary = (
                sdf_starts_patch_all[:, -1:, :] - sdf_starts_patch_all[:, -2:-1, :]
            ) / dist_y[:, -1:, :]

            df_dx = torch.cat([df_dx_left_boundary, df_dx, df_dx_right_boundary], dim=2)
            df_dy = torch.cat([df_dy_top_boundary, df_dy, df_dy_bottom_boundary], dim=1)

            projected_gradients = torch.cat([df_dt, df_dx, df_dy], dim=-1)
            V_inverse = V_inverse_patch_all[patch_indices]
            gradients = (V_inverse @ projected_gradients[..., None])[..., 0]

        elif gradient_method == "ad":
            gradients = self.sdf_network.gradient(
                positions_starts_patch_all.reshape(-1, 3)
            ).reshape(num_samples, patch_H, patch_W, 3)

        elif gradient_method == "fd":
            self.fd_epsilon = 1e-3
            eps = self.fd_epsilon
            p_base = positions_starts_patch_all
            offsets = torch.tensor(
                [
                    [-eps, 0.0, 0.0],
                    [eps, 0.0, 0.0],
                    [0.0, -eps, 0.0],
                    [0.0, eps, 0.0],
                    [0.0, 0.0, -eps],
                    [0.0, 0.0, eps],
                ],
                device=device,
                dtype=p_base.dtype,
            )
            positions_concat = (
                p_base.unsqueeze(0) + offsets[:, None, None, None, :]
            ).reshape(-1, 3)

            sdf_concat = self._eval_sdf_chunked(
                positions_concat, chunk_size=262144, use_checkpoint=use_ckpt
            ).reshape(6, num_samples, patch_H, patch_W, 1)
            df_dx = (sdf_concat[1] - sdf_concat[0]) / (2.0 * eps)
            df_dy = (sdf_concat[3] - sdf_concat[2]) / (2.0 * eps)
            df_dz = (sdf_concat[5] - sdf_concat[4]) / (2.0 * eps)
            gradients = torch.cat([df_dx, df_dy, df_dz], dim=-1)
        else:
            raise ValueError(f"Unknown gradient_method: {gradient_method}")

        weights_sum = accumulate_along_rays_patch_based(
            weights, patch_indices, n_patches=num_patch
        ).reshape(num_patch, patch_H, patch_W, 1)

        comp_normal = accumulate_along_rays_patch_based(
            weights,
            patch_indices,
            values=gradients.reshape(num_samples, patch_H * patch_W, 3),
            n_patches=num_patch,
        ).reshape(num_patch, patch_H, patch_W, 3)

        inv_s = self.deviation_network(
            torch.zeros([1, 3], device=device)
        )[:, :1].clip(1e-6, 1e6)

        return {
            "s_val": 1.0 / inv_s,
            "weight_sum": weights_sum,
            "gradients": gradients,
            "comp_normal": comp_normal,
            "samples_per_ray": samples_per_ray,
        }

    @torch.no_grad()
    def render_normal_pixel_based(
        self, rays_o: Tensor, rays_d: Tensor, near: Tensor, far: Tensor
    ) -> Tuple[Tensor, Tensor]:
        device = rays_o.device

        def alpha_fn(
            t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor, ret_sdf: bool = False
        ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions_starts = t_origins + t_dirs * t_starts
            positions_ends = t_origins + t_dirs * t_ends

            t_starts_shift_left = torch.cat([t_starts[1:], t_starts[-1:]], dim=0)
            diff_mask = ((t_ends - t_starts_shift_left) != 0).squeeze(-1)
            positions_ends_diff = positions_ends[diff_mask].reshape(-1, 3)

            positions_all = torch.cat([positions_starts, positions_ends_diff], dim=0)

            sdf_all = self._eval_sdf_chunked(positions_all, use_checkpoint=False)
            sdf_start = sdf_all[: positions_starts.shape[0]]
            sdf_end_diff = sdf_all[positions_starts.shape[0] :]

            sdf_start_shift_left = torch.cat([sdf_start[1:], sdf_start[-1:]], dim=0).clone()
            sdf_start_shift_left[diff_mask] = sdf_end_diff

            inv_s = self.deviation_network(
                torch.zeros([1, 3], device=device)
            )[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf_start.shape[0], 1)

            prev_cdf = torch.sigmoid(sdf_start * inv_s)
            next_cdf = torch.sigmoid(sdf_start_shift_left * inv_s)

            p = prev_cdf - next_cdf
            c = prev_cdf

            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            if ret_sdf:
                return alpha, sdf_start, sdf_start_shift_left
            return alpha

        ray_indices, t_starts, t_ends = ray_marching(
            rays_o,
            rays_d,
            t_min=near.squeeze(),
            t_max=far.squeeze(),
            grid=self.occupancy_grid,
            render_step_size=float(self.sampling_step_size),
            stratified=True,
            cone_angle=0.0,
            alpha_thre=0.0,
            early_stop_eps=1e-3,
            alpha_fn=alpha_fn,
        )

        n_rays = rays_o.shape[0]
        if ray_indices.numel() == 0:
            return (
                torch.zeros((n_rays, 3), device=device, dtype=rays_o.dtype),
                torch.zeros((n_rays, 1), device=device, dtype=rays_o.dtype),
            )

        alpha = alpha_fn(t_starts, t_ends, ray_indices)

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * midpoints
        gradients = self.sdf_network.gradient(positions).reshape(-1, 3)

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        comp_normal = accumulate_along_rays(
            weights, ray_indices, values=gradients, n_rays=n_rays
        )
        comp_depth = accumulate_along_rays(
            weights, ray_indices, values=midpoints, n_rays=n_rays
        )
        return comp_normal, comp_depth

    def extract_geometry(
        self, bound_min: Tensor, bound_max: Tensor, resolution: int, threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        return extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: -self.sdf_network.sdf(pts),
            device=self.device,
        )