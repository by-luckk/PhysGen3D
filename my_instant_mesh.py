import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

from submodules.instant_mesh.utils.train_util import instantiate_from_config
from submodules.instant_mesh.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from submodules.instant_mesh.utils.mesh_util import save_obj, save_obj_with_mtl
from submodules.instant_mesh.utils.infer_util import resize_foreground


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    return frames

class MyInstantMesh:
    def __init__(
        self,
        config,
        input_path,
        output_path='outputs/',
        seed=42,
    ):
        seed_everything(seed)
        self.config = OmegaConf.load(config)
        self.config_name = os.path.basename(config).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config
        self.IS_FLEXICUBES = True if self.config_name.startswith('instant-mesh') else False
        self.device = torch.device('cuda')

        # load diffusion model
        print('Loading diffusion model ...')
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.infer_config.zero123plus_v12, 
            custom_pipeline=self.infer_config.zero123plus,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # load custom white-background UNet
        print('Loading custom white-background unet ...')
        if os.path.exists(self.infer_config.unet_path):
            unet_ckpt_path = self.infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)

        self.pipeline = self.pipeline.to(self.device)

        # load reconstruction model
        print('Loading reconstruction model ...')
        self.model = instantiate_from_config(self.model_config)
        if os.path.exists(self.infer_config.model_path):
            model_ckpt_path = self.infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{self.config_name.replace('-', '_')}.ckpt", repo_type="model")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)

        self.model = self.model.to(self.device)
        if self.IS_FLEXICUBES:
            self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

        # make output directories
        self.image_path = os.path.join(output_path, 'images')
        self.mesh_path = os.path.join(output_path, 'meshes')
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        # self.video_path = os.path.join(output_path, 'videos')
        # os.makedirs(self.image_path, exist_ok=True)

        # process input files
        if os.path.isdir(input_path):
            self.input_files = [
                os.path.join(input_path, file) 
                for file in os.listdir(input_path) 
                if (file.endswith('.png') or file.endswith('.jpg') or file.endswith('.webp')) and not file.endswith('black.jpg')
            ]
        else:
            self.input_files = [input_path]
        print(f'Total number of input images: {len(self.input_files)}')
    
    def add_white_border(self, input_image, min_border=30):
        width, height = input_image.size
        if width > height:
            new_size = width + 2 * min_border
            padding = (min_border, min_border, 
                    (new_size - height) // 2, (new_size - height) // 2)
        else:
            new_size = height + 2 * min_border
            padding = ((new_size - width) // 2, (new_size - width) // 2, 
                    min_border, min_border)
        bordered_image = ImageOps.expand(input_image, border=padding, fill='white')

        return bordered_image

    def multiview_generation(self, diffusion_steps=75):
        self.outputs = []
        for idx, image_file in enumerate(self.input_files):
            name = os.path.basename(image_file).split('.')[0]
            print(f'[{idx+1}/{len(self.input_files)}] Imagining {name} ...')
            input_image = Image.open(image_file)
            input_image = self.add_white_border(input_image, min_border=100)
            
            # sampling
            output_image = self.pipeline(
                input_image, 
                num_inference_steps=diffusion_steps, 
            ).images[0]

            output_image.save(os.path.join(self.image_path, f'{name}.png'))
            print(f"Image saved to {os.path.join(self.image_path, f'{name}.png')}")

            images = np.asarray(output_image, dtype=np.float32) / 255.0
            images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
            images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)

            self.outputs.append({'name': name, 'images': images})

        # delete pipeline to save memory
        del self.pipeline

    def reconstruction(self, scale=1.0, view=6, distance=4.5, export_texmap=False, save_video=False):
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*scale).to(self.device)
        chunk_size = 20 if self.IS_FLEXICUBES else 1

        for idx, sample in enumerate(self.outputs):
            name = sample['name']
            print(f'[{idx+1}/{len(self.outputs)}] Creating {name} ...')

            images = sample['images'].unsqueeze(0).to(self.device)
            images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

            if view == 4:
                indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
                images = images[:, indices]
                input_cameras = input_cameras[:, indices]

            with torch.no_grad():
                # get triplane
                planes = self.model.forward_planes(images, input_cameras)

                # get mesh
                mesh_path_idx = os.path.join(self.mesh_path, f'{name}.obj')

                mesh_out = self.model.extract_mesh(
                    planes,
                    use_texture_map=export_texmap,
                    **self.infer_config,
                )
                if export_texmap:
                    vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
                    save_obj_with_mtl(
                        vertices.data.cpu().numpy(),
                        uvs.data.cpu().numpy(),
                        faces.data.cpu().numpy(),
                        mesh_tex_idx.data.cpu().numpy(),
                        tex_map.permute(1, 2, 0).data.cpu().numpy(),
                        mesh_path_idx,
                    )
                else:
                    vertices, faces, vertex_colors = mesh_out
                    save_obj(vertices, faces, vertex_colors, mesh_path_idx)
                print(f"Mesh saved to {mesh_path_idx}")

                # # get video
                # if save_video:
                #     video_path_idx = os.path.join(self.video_path, f'{name}.mp4')
                #     render_size = self.infer_config.render_resolution
                #     render_cameras = get_render_cameras(
                #         batch_size=1, 
                #         M=120, 
                #         radius=distance, 
                #         elevation=20.0,
                #         is_flexicubes=self.IS_FLEXICUBES,
                #     ).to(self.device)
                    
                #     frames = render_frames(
                #         self.model, 
                #         planes, 
                #         render_cameras=render_cameras, 
                #         render_size=render_size, 
                #         chunk_size=chunk_size, 
                #         is_flexicubes=self.IS_FLEXICUBES,
                #     )

                #     save_video(
                #         frames,
                #         video_path_idx,
                #         fps=30,
                #     )
                #     print(f"Video saved to {video_path_idx}")
