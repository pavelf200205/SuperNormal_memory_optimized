import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
import pyexr
import open3d as o3d
import time


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        normal_dir = conf.get_string('normal_dir')

        self.data_dir = conf.get_string('data_dir')
        self.cameras_name = conf.get_string('cameras_name')
        self.exclude_view_list = conf['exclude_views']
        self.upsample_factor = conf.get_int('upsample_factor', default=1)
        ic(self.exclude_view_list)

        mesh_path = os.path.join(self.data_dir, 'mesh_Gt.ply')
        if os.path.exists(mesh_path):
            self.mesh_gt = o3d.io.read_triangle_mesh(mesh_path)
        else:
            self.mesh_gt = None
        self.points_gt = None

        camera_dict = np.load(os.path.join(self.data_dir, self.cameras_name))
        self.camera_dict = camera_dict
        self.normal_lis = sorted(glob(os.path.join(self.data_dir, normal_dir, '*.exr')))
        self.n_images = len(self.normal_lis)
        self.train_images = set(range(self.n_images)) - set(self.exclude_view_list)
        self.img_idx_list = [int(os.path.basename(x).split('.')[0]) for x in self.normal_lis]

        # Get target dimensions from the first image
        first_img = pyexr.read(self.normal_lis[0])[..., :3]
        self.H = int(first_img.shape[0] * self.upsample_factor)
        self.W = int(first_img.shape[1] * self.upsample_factor)

        print("loading normal maps...")
        
        # SYSTEM RAM OPTIMIZATION 4: PRE-ALLOCATION
        # Create the final tensor in RAM instantly, bypassing the massive np.stack memory spike
        self.normals = torch.empty((self.n_images, self.H, self.W, 3), dtype=torch.float16)

        for i, im_name in enumerate(self.normal_lis):
            img_np = pyexr.read(im_name)[..., :3].astype(np.float32)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0) # 1, 3, H, W
            
            if self.upsample_factor > 1:
                img_t = F.interpolate(img_t, scale_factor=self.upsample_factor, mode='bilinear', align_corners=False)
            
            # Drop straight into pre-allocated memory as float16
            self.normals[i] = img_t.squeeze(0).permute(1, 2, 0).half()

        print("loading normal maps done.")

        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks = torch.empty((self.n_images, self.H, self.W), dtype=torch.bool)

        for i, im_name in enumerate(self.masks_lis):
            img_np = cv.imread(im_name)[..., 0] / 255.0
            img_t = torch.from_numpy(img_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            
            if self.upsample_factor > 1:
                img_t = F.interpolate(img_t, scale_factor=self.upsample_factor, mode='nearest')
                
            self.masks[i] = img_t.squeeze(0).squeeze(0).bool()

        self.total_pixel = self.masks.sum().item()

        # Set background of normal map to 0 using the pure tensor
        self.normals[~self.masks] = 0

        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.img_idx_list]

        self.intrinsics_all = []
        self.pose_all = []

        # Zip loop cleaned up - normal and mask were never used inside this block
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            if self.upsample_factor > 1:
                intrinsics[0, 0] *= self.upsample_factor
                intrinsics[1, 1] *= self.upsample_factor
                intrinsics[0, 2] *= self.upsample_factor
                intrinsics[1, 2] *= self.upsample_factor
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal_length = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        self.image_pixels = self.H * self.W

        self.object_bbox_min = np.array([-1., -1., -1.])
        self.object_bbox_max = np.array([1.,  1.,  1.])
        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1, within_mask=False):
        # Dynamically extract and format the bool tensor back to numpy just for this specific call
        mask_np = self.masks[img_idx].numpy().astype(np.uint8) * 255
        mask_np = cv.resize(mask_np, (int(self.W // resolution_level), int(self.H // resolution_level)), interpolation=cv.INTER_NEAREST).astype(bool)

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, int(self.W // l), device=self.device)
        ty = torch.linspace(0, self.H - 1, int(self.H // l), device=self.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)
        rays_o = rays_o.transpose(0, 1)
        rays_v = rays_v.transpose(0, 1)

        if within_mask:
            return rays_o[mask_np], rays_v[mask_np]
        else:
            return rays_o, rays_v

    def gen_patches_at(self, img_idx, resolution_level=1, patch_H=3, patch_W=3):
        tx = torch.linspace(0, self.W - 1, int(self.W // resolution_level), device=self.device)
        ty = torch.linspace(0, self.H - 1, int(self.H // resolution_level), device=self.device)
        pixels_y, pixels_x = torch.meshgrid(ty, tx)

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, :3, :3], p[..., None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, :3, :3], rays_v[:, :, :, None]).squeeze()

        rays_right = self.pose_all[img_idx, :3, 0].expand(rays_v.shape)
        rays_down = self.pose_all[img_idx, :3, 1].expand(rays_v.shape)
        V_concat = torch.cat([rays_v[..., None, :], rays_right[..., None, :], rays_down[..., None, :]], dim=-2)
        # VRAM optimization: invert only extracted patches instead of full (H, W, 3, 3) image

        height, width, _ = rays_v.shape
        horizontal_num_patch = width // patch_W
        vertical_num_patch = height // patch_H
        rays_v_patches_all = []
        rays_V_inverse_patches_all = []
        
        for i in range(0, height-patch_H//2-1, patch_H):
            for j in range(0, width-patch_W//2-1, patch_W):
                rays_v_patch = rays_v[i:i + patch_H, j:j + patch_W]
                rays_v_patches_all.append(rays_v_patch)

                V_patch = V_concat[i:i + patch_H, j:j + patch_W]
                rays_V_inverse_patches_all.append(torch.inverse(V_patch))
                
        rays_v_patches_all = torch.stack(rays_v_patches_all, dim=0)
        rays_V_inverse_patches_all = torch.stack(rays_V_inverse_patches_all, dim=0)  
        
        rays_o_patches_all = self.pose_all[img_idx, :3, 3].expand(rays_v_patches_all.shape)  

        rays_o_patch_center = rays_o_patches_all[:, patch_H//2, patch_W//2]  
        rays_d_patch_center = rays_v_patches_all[:, patch_H//2, patch_W//2]  

        marching_plane_normal_patches_all = self.pose_all[img_idx, :3, 2].expand(rays_d_patch_center.shape)  

        return rays_o_patch_center, \
                rays_d_patch_center, \
            rays_o_patches_all, \
            rays_v_patches_all, \
            marching_plane_normal_patches_all, \
            rays_V_inverse_patches_all, horizontal_num_patch, vertical_num_patch

    def gen_random_patches(self, num_patch, patch_H=3, patch_W=3):
        patch_center_x = torch.randint(low=0+patch_W//2, high=self.W-1-patch_W//2, size=[num_patch], device='cpu')  
        patch_center_y = torch.randint(low=0+patch_H//2, high=self.H-1-patch_H//2, size=[num_patch], device='cpu')  

        patch_center_x_all = patch_center_x[:, None, None] + torch.arange(-patch_W//2+1, patch_W//2+1, device='cpu').repeat(patch_H, 1)   
        patch_center_y_all = patch_center_y[:, None, None] + torch.arange(-patch_H//2+1, patch_H//2+1, device='cpu').reshape(-1, 1).repeat(1, patch_W)   

        img_idx = np.random.choice(list(self.train_images), size=[num_patch])  
        img_idx = torch.tensor(img_idx, device='cpu')
        img_idx_expand = img_idx.view(-1, 1, 1).expand_as(patch_center_x_all)  

        normal = self.normals[img_idx_expand, patch_center_y_all, patch_center_x_all].to(self.device).float()  
        mask = self.masks[img_idx_expand, patch_center_y_all, patch_center_x_all].unsqueeze(-1).to(self.device).float()     

        patch_center_x_all_gpu = patch_center_x_all.to(self.device)
        patch_center_y_all_gpu = patch_center_y_all.to(self.device)
        img_idx_expand_gpu = img_idx_expand.to(self.device)
        img_idx_gpu = img_idx.to(self.device)

        p_all = torch.stack([patch_center_x_all_gpu, patch_center_y_all_gpu, torch.ones_like(patch_center_y_all_gpu)], dim=-1).float()  
        p_all = torch.matmul(self.intrinsics_all_inv[img_idx_expand_gpu, :3, :3], p_all[..., None])[..., 0]  
        p_norm_all = torch.linalg.norm(p_all, ord=2, dim=-1, keepdim=True)  
        rays_d_patch_all = p_all / p_norm_all  
        rays_d_patch_all = torch.matmul(self.pose_all[img_idx_gpu, None, None, :3, :3], rays_d_patch_all[..., None])[..., 0]  
        rays_o_patch_all = self.pose_all[img_idx_gpu, None, None, :3, 3].expand(rays_d_patch_all.shape)  

        rays_right = self.pose_all[img_idx_gpu, None, None, :3, 0].expand(rays_d_patch_all.shape)
        rays_down = self.pose_all[img_idx_gpu, None, None, :3, 1].expand(rays_d_patch_all.shape)
        V_concat = torch.cat([rays_d_patch_all[..., None, :],
                              rays_right[..., None, :],
                              rays_down[..., None, :]], dim=-2)
        V_inverse_patch_all = torch.inverse(V_concat)

        marching_plane_normal = self.pose_all[img_idx_gpu, :3, 2].expand((num_patch, 3))  

        return rays_o_patch_all, \
                rays_d_patch_all, \
                marching_plane_normal, \
                V_inverse_patch_all, \
                normal,\
                mask

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        c = torch.sum(rays_o**2, dim=-1, keepdim=True) - 1.0
        mid = 0.5 * (-b) / a
        near = mid - torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        far = mid + torch.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        return near[..., 0], far[..., 0]

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)