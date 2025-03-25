# SPDX-License-Identifier: Apache-2.0
import sys
import argparse
from pathlib import Path
import open3d as o3d
import mitsuba as mi
import drjit as dr
import numpy as np
import math


def extract_cam_from_txt(cam_file):
    """Extract camera parameters from camera txt files as Mitsuba expects them"""

    def compute_principal_pt(calib_pt, dim):
        '''Convert principal point to format expected by Mitsuba - offset from center'''
        dim_div_2 = dim/2.0
        return (dim_div_2 - calib_pt) / dim

    params = np.loadtxt(cam_file)
    K, R, t, (width, height, channels) = params[:3], params[3:6], params[6], params[7].astype(int)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    cam_xform = np.linalg.inv(T)
    cam_xform = np.matmul(cam_xform, np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float32))
    fovx = 2 * np.arctan(0.5 * width / K[0,0])
    fovx = (180.0 / math.pi) * fovx.item()
    principle_x = compute_principal_pt(K[0,2], width.item())
    principle_y = compute_principal_pt(K[1,2], height.item())
    return (cam_xform, fovx, width.item(), height.item(), (principle_x, principle_y))


def load_reference_image(img_file):
    """Loads an image and returns it as a linearized float Mitsuba Tensor"""
    bmp = mi.Bitmap(str(img_file))
    bmp = bmp.convert(component_format=mi.Struct.Type.Float32, srgb_gamma=False)
    return mi.TensorXf(bmp)


def prepare_neus_model(dataset_dir, prepared_mesh, tex_size):
    '''Prepare mesh by generating normals and a UV mapping'''
    # Load mesh either from a ply or npz file
    mesh_path = dataset_dir / 'neus_mesh.ply'
    mesh_npz = dataset_dir / 'mesh.npz'
    if mesh_path.exists():
        print(f'Creating prepared mesh for {mesh_path}...')
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    elif mesh_npz.exists():
        print(f'Creating prepared mesh for {mesh_npz}...')
        mesh_npz = np.load(mesh_npz)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh_npz['vertices']),
                                         o3d.utility.Vector3iVector(mesh_npz['triangles']))
        mesh.vertex_normals = o3d.utility.Vector3dVector(mesh_npz['vertex_normals'])
    else:
        print('No mesh found!')
        sys.exit(1)

    print('Cleaning mesh...')
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertex_normals) == 0:
        print('Computing normals...')
        mesh.compute_vertex_normals()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    if 'normals' in tmesh.triangle: # triangle normals are by product of computing vertex normals
        del tmesh.triangle.normals

    print('Computing UVs...')
    tmesh.compute_uvatlas(tex_size)

    print(tmesh)

    # Save mesh and its uvmask
    o3d.t.io.write_triangle_mesh(str(prepared_mesh), tmesh)
    print(f'Prepared mesh written to {prepared_mesh}')
    return tmesh


def save_image(img, name, output_dir):
    """Saves a float image array with range [0..1] as 8 bit PNG"""
    # scale to 0-255
    texture = o3d.core.Tensor(img * 255.0).to(o3d.core.Dtype.UInt8)
    texture = o3d.t.geometry.Image(texture)
    o3d.t.io.write_image(str(output_dir/name), texture)


def update_sensor_view(params, cam):
    cam_xform, fov, _, _, _  = cam
    params['sensor.to_world'] = cam_xform
    params['sensor.x_fov'] = fov
    params.update()


def make_scene(mesh, cam_xform, fov, width, height, principle_pts):
    t_from_np = mi.ScalarTransform4f(cam_xform)
    env_t = mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=90).rotate(axis=[1,0,0], angle=90)
    scene_dict = {
        "type": "scene",
        "integrator": {'type': 'path'},
        "light": {
            "type": "envmap",
             "to_world": env_t,
             },
        "sensor": {
            "type": "perspective",
            "fov": fov,
            "to_world": t_from_np,
            "principal_point_offset_x": principle_pts[0],
            "principal_point_offset_y": principle_pts[1],
            "thefilm": {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                },
            "thesampler": {
                "type": "multijitter",
                "sample_count": 64,
                },
            },
        "neus": mesh,
        }

    # Put some random data in the environment map
    env_width = 256
    scene_dict['light']['bitmap'] = mi.Bitmap(array=np.ones((int(env_width/2), env_width, 3)), pixel_format=mi.Bitmap.PixelFormat.RGB)

    scene = mi.load_dict(scene_dict)
    return scene


def run_estimation(mesh, cameras, images, iterations, batch_size, with_metallic=True, with_roughness=True, env_width=256):
    mesh_opt = mesh.to_mitsuba('neus')
    cam = cameras[0]
    scene = make_scene(mesh_opt, cam[0], cam[1], cam[2], cam[3], cam[4])

    params = mi.traverse(scene)
    empty_envmap = np.ones((int(env_width/2), env_width, 3))
    params['light.data'] = empty_envmap
    params.update()

    # Loss fucntion
    def mse(image, ref_img):
        return dr.mean(dr.sqr(image - ref_img))

    print(params)
    bsdf_key_prefix = 'neus.bsdf.'

    # Setup parameters for optimization
    opt = mi.ad.Adam(lr=0.05, mask_updates=False)
    opt[bsdf_key_prefix + 'base_color.data'] = params[bsdf_key_prefix + 'base_color.data']
    if with_roughness:
        opt[bsdf_key_prefix + 'roughness.data'] = params[bsdf_key_prefix + 'roughness.data']
    if with_metallic:
        opt[bsdf_key_prefix + 'metallic.data'] = params[bsdf_key_prefix + 'metallic.data']
    opt['light.data'] = params['light.data']
    params.update(opt)

    integrator = mi.load_dict({'type': 'prb'})
    for i in range(iterations):
        for b in range(batch_size):
            # Select a reference image
            ref_idx = np.random.randint(len(cameras))
            ref_img = images[ref_idx]

            # Update scene camera for current input image
            update_sensor_view(params, cameras[ref_idx])

            # Render
            img = mi.render(scene, params, spp=8, seed = i*b, integrator=integrator)

            # Compute loss and back propogate
            loss = mse(img, ref_img)
            dr.backward(loss)

        opt.step()
        opt[bsdf_key_prefix + 'base_color.data'] = dr.clamp(opt[bsdf_key_prefix + 'base_color.data'], 0.0, 1.0)
        if with_roughness:
            opt[bsdf_key_prefix + 'roughness.data'] = dr.clamp(opt[bsdf_key_prefix + 'roughness.data'], 0.0, 1.0)
        if with_metallic:
            opt[bsdf_key_prefix + 'metallic.data'] = dr.clamp(opt[bsdf_key_prefix + 'metallic.data'], 0.0, 1.0)
        params.update(opt)
        print(f'Iteration {i} complete')

    # Gather results to return
    albedo_img = params[bsdf_key_prefix + 'base_color.data'].numpy()
    if with_roughness:
        roughness_img = params[bsdf_key_prefix + 'roughness.data'].numpy()
    else:
        roughness_img = params[bsdf_key_prefix + 'roughness.value'].numpy()
    if with_metallic:
        metallic_img = params[bsdf_key_prefix + 'metallic.data'].numpy()
    else:
        metallic_img = params[bsdf_key_prefix + 'metallic.value'].numpy()
    envmap_img = params['light.data'].numpy()

    return (albedo_img, roughness_img, metallic_img, envmap_img)


def load_input_model(model_path, with_metallic, with_roughness, tex_dim=2048, starting_values=(0.5, 0.0, 0.0)):
    print(f'Loading {model_path}...')
    mesh = o3d.t.io.read_triangle_mesh(str(model_path))
    mesh.material.set_default_properties()
    mesh.material.material_name = 'defaultLit' # note: ignored by Mitsuba, just used to visualize in Open3D

    # Start with either empty maps or maps from intermediate results
    print(starting_values)
    albedo_start, roughness_start, metallic_start = starting_values
    mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(albedo_start + np.zeros((tex_dim,tex_dim,3), dtype=np.float32))
    if with_roughness:
        mesh.material.texture_maps['roughness'] = o3d.t.geometry.Image(roughness_start + np.zeros((tex_dim,tex_dim,1), dtype=np.float32))
    else:
        mesh.material.scalar_properties['roughness'] = roughness_start

    if with_metallic:
        mesh.material.texture_maps['metallic'] = o3d.t.geometry.Image(metallic_start + np.zeros((tex_dim,tex_dim,1), dtype=np.float32))
    else:
        mesh.material.scalar_properties['metallic'] = metallic_start

    return mesh


def load_input_data(input_dir):
    print(f'Loading input data from {input_dir}...')
    use_npz_files = False
    cam_inputs = sorted(list(input_dir.glob('camera*.txt')))
    n_inputs = len(cam_inputs)
    if n_inputs == 0:
        print('No camera txt files found. Looking for NPZ camera files')
        cam_inputs = sorted(list(input_dir.glob('camera*.npz')))
        n_inputs = len(cam_inputs)
        use_npz_files = True

    print(f'Found {n_inputs} input images')
    cameras = []
    for c in cam_inputs:
        if use_npz_files:
            cameras.append(extract_cam_from_npz(c))
        else:
            cameras.append(extract_cam_from_txt(c))

    ref_images = []
    img_inputs = sorted(list(input_dir.glob('image*.png')))
    for i in img_inputs:
        ref_images.append(load_reference_image(i))

    return (cameras, ref_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Script that estimates texture maps and optionally environment from input images and geometry.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_dir', type=Path, help="Path to test directory of dataset to process")
    parser.add_argument('--env-width', type=int, default=128)
    parser.add_argument('--tex-width', type=int, default=2048, help="The dimensions of the texture")
    parser.add_argument('--starting-albedo', type=float, default=0.0)
    parser.add_argument('--starting-roughness', type=float, default=0.5)
    parser.add_argument('--starting-metallic', type=float, default=0.0)
    parser.add_argument('--with-metallic', action='store_true', help="Estimate metallic texture")
    parser.add_argument('--no-roughness', action='store_true', help="Don't estimate roughness texture")
    parser.add_argument('--with-cpu', action='store_true', help="Run Mitsuba on CPU instead of CUDA")
    parser.add_argument('--iterations', type=int, default=40, help="Number of iterations")
    parser.add_argument('--batch-size', type=int, default=8, help="Number of iterations")

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print(args)

    # Initialize Mitsuba
    if args.with_cpu:
        mi.set_variant('llvm_ad_rgb')
    else:
        mi.set_variant('cuda_ad_rgb')

    dataset_dir = args.dataset_dir
    if not dataset_dir.exists():
        print(f'{dataset_dir} does not exist!')
        sys.exit(1)
    input_dir = dataset_dir / 'inputs'
    with_metallic = args.with_metallic
    with_roughness = not args.no_roughness
    tex_width = args.tex_width

    # Check for prepared mesh
    prepared_mesh_path = dataset_dir / 'prepared_mesh.obj'
    if not prepared_mesh_path.exists():
        prepare_neus_model(dataset_dir, prepared_mesh_path, tex_width)

    mesh= load_input_model(prepared_mesh_path, with_metallic, with_roughness, tex_width, (args.starting_albedo, args.starting_roughness, args.starting_metallic))
    input_cameras, input_images = load_input_data(input_dir)
    envmap_path = dataset_dir / 'env_512_rotated.hdr'

    # Estimate material maps
    print('Running material estimation...', flush=True)
    albedo, roughness, metallic, envmap = run_estimation(mesh,
                                                         input_cameras,
                                                         input_images,
                                                         args.iterations,
                                                         args.batch_size,
                                                         env_width=args.env_width,
                                                         with_metallic=with_metallic,
                                                         with_roughness=with_roughness)

    # Save maps
    print('Saving final results...')
    save_image(albedo, 'predicted_albedo.png', dataset_dir)
    if with_roughness:
        save_image(roughness, 'predicted_roughness.png', dataset_dir)
    if with_metallic:
        save_image(metallic, 'predicted_metallic.png', dataset_dir)
    mi.Bitmap(envmap).write(str(dataset_dir/'predicted_envmap.exr'))

    # Save everything to NPZ file
    mesh_vertices = mesh.vertex.positions.numpy()
    mesh_normals = mesh.vertex.normals.numpy()
    mesh_triangles = mesh.triangle.indices.numpy()
    mesh_uvs = mesh.triangle.texture_uvs.numpy()
    out_data = {
        'vertices': mesh_vertices,
        'triangles': mesh_triangles,
        'vertex_normals': mesh_normals,
        'uvmap': mesh_uvs,
        'envmap': envmap,
        'tex_albedo': albedo,
    }
    if with_roughness:
        out_data['tex_roughness'] = roughness
    if with_metallic:
        out_data['tex_metallic'] = metallic
    else:
        out_data['metallic'] = metallic[0,0]
    np.savez(dataset_dir/'estimated_materials.npz', **out_data)
