import argparse
import os
import torch
import copy
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="path to the model weights")
    parser.add_argument("--input_images", type=str, nargs='+', required=True, help="list of input image paths")
    parser.add_argument("--output_dir", type=str, required=True, help="directory to save the output 3D model")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--schedule", type=str, default='linear', choices=["linear", "cosine"], help="schedule for global alignment")
    parser.add_argument("--niter", type=int, default=300, help="number of iterations for global alignment")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="minimum confidence threshold")
    parser.add_argument("--as_pointcloud", action='store_true', help="output as point cloud instead of mesh")
    parser.add_argument("--mask_sky", action='store_true', help="mask sky in output")
    parser.add_argument("--clean_depth", action='store_true', help="clean depth maps in output")
    parser.add_argument("--transparent_cams", action='store_true', help="render cameras as transparent")
    parser.add_argument("--cam_size", type=float, default=0.05, help="camera size in the output scene")
    return parser

def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, name='scene'):
    # Function to convert the scene output to a GLB file
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    mask = to_numpy(mask)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    
    # Full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # Add each camera
    # for i, pose_c2w in enumerate(cams2world):
    #     camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
    #     add_scene_cam(scene, pose_c2w, camera_edge_color,
    #                   None if transparent_cams else imgs[i], focals[i],
    #                   imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    # rot = np.eye(4)
    # rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    # scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, f'{name}.ply')
    if not silent:
        print(f'(exporting 3D scene to {outfile})')
    scene.export(file_obj=outfile)
    return outfile

def run_dust3r(weights, input_path, output_dir, device='cuda', image_size=512, schedule='linear',
                  niter=300, min_conf_thr=3.0, as_pointcloud=False, mask_sky=True,
                  clean_depth=False, transparent_cams=False, cam_size=0.05):
    os.makedirs(output_dir, exist_ok=True)
    # Load images
    imgs = load_images(input_path, size=image_size, square_ok=True)
    img_names = [path.split('/')[-1].split('.')[0] for path in input_path]
    model = AsymmetricCroCo3DStereo.from_pretrained(weights).to(device)

    for img, name in zip(imgs, img_names):
        img_pair = [img, copy.deepcopy(img)]
        img_pair[0]['idx'] = 0
        img_pair[1]['idx'] = 1

        # Create pairs and run inference
        pairs = make_pairs(img_pair, scene_graph='complete', prefilter=None, symmetrize=True)
        
        output = inference(pairs, model, device, batch_size=1, verbose=True)

        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)

        if clean_depth:
            scene = scene.clean_pointcloud()
        # if mask_sky:
        #     scene = scene.mask_sky()
        scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
        # Convert scene to PLY and save
        _convert_scene_output_to_glb(output_dir, scene.imgs, scene.get_pts3d(), scene.get_masks(),
                                            scene.get_focals().cpu(), scene.get_im_poses().cpu(),
                                            cam_size=cam_size, as_pointcloud=as_pointcloud,
                                            transparent_cams=transparent_cams, name=name)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # Run the inference and generate the output model
    run_dust3r(
        weights=args.weights,
        input_path=args.input_images,
        output_dir=args.output_dir,
        device=args.device,
        image_size=args.image_size,
        schedule=args.schedule,
        niter=args.niter,
        min_conf_thr=args.min_conf_thr,
        as_pointcloud=args.as_pointcloud,
        mask_sky=args.mask_sky,
        clean_depth=args.clean_depth,
        transparent_cams=args.transparent_cams,
        cam_size=args.cam_size
    )
