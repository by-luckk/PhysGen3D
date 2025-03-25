import mitsuba as mi
import drjit as dr
import os
import json
import numpy as np
import argparse
import sys
import cv2
import open3d as o3d
from PIL import Image
from scipy.spatial import KDTree
import trimesh
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
from engine.particle_io import ParticleIO

# Initialize Mitsuba
mi.set_variant('cuda_ad_rgb')

SIZE = 2
center = [SIZE/2, SIZE/2, SIZE/2]
scale = 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--begin', type=int, default=0, help='Beginning frame')
    parser.add_argument('-e', '--end', type=int, default=10000, help='Ending frame')
    parser.add_argument('-s', '--step', type=int, default=1, help='Frame step')
    parser.add_argument('-g', '--gui', action='store_true', help='Show GUI')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('-i', '--in-dir', type=str, help='Input folder')
    parser.add_argument('-v', '--video', type=str, help='Generate video')
    parser.add_argument('-t', '--shutter-time', type=float, default=2e-3, help='Shutter time')
    parser.add_argument('-f', '--force', action='store_true', help='Overwrite existing outputs')
    parser.add_argument('-p', '--fps', type=int, help='Frame per second')
    parser.add_argument('--path', type=str, help='Input path')
    parser.add_argument('--name', type=str, help='Object name')
    parser.add_argument('--env', type=str, help='Environment light file name')
    parser.add_argument('-M', '--max-particles', type=int, default=128, help='Max num particles (million)')
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()
output_folder = args.out_dir
os.makedirs(output_folder, exist_ok=True)

def load_input_model(model_path, with_metallic=False, with_roughness=False, transform=None,
                     tex_map=None, tex_dim=2048, starting_values=(0.5, 1, 0)):
    print(f'Loading {model_path}...')
    albedo_start, roughness_start, metallic_start = starting_values 

    # load mesh (with texture)   
    mesh_old = o3d.io.read_triangle_mesh(str(model_path), False)
    print(f"Number of vertices in mesh_old: {len(mesh_old.vertices)}")
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_old)
    print(f"Number of vertices in mesh: {mesh.vertex['positions'].shape[0]}")

    mesh.material.material_name = 'defaultLit' # note: ignored by Mitsuba, just used to visualize in Open3D

    # Load texture maps
    if tex_map is not None:
        # albedo_image = np.array(Image.open(tex_map)) / 255.0
        mesh.material.texture_maps['albedo'] = o3d.t.io.read_image(tex_map)
        tex_dim = len(np.array(mesh.material.texture_maps['albedo']))
        print('Given albedo texture', mesh.material.texture_maps['albedo'])
    elif 'albedo' in mesh.material.texture_maps:
        tex_dim = len(np.array(mesh.material.texture_maps['albedo']))
        print('Loaded albedo texture', mesh.material.texture_maps['albedo'])
    else:
        mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(albedo_start + np.zeros((tex_dim, tex_dim, 3), dtype=np.float32))
        print('Default albedo texture', mesh.material.texture_maps['albedo'])
        
    if with_roughness:
        mesh.material.texture_maps['roughness'] = o3d.t.geometry.Image(roughness_start + np.zeros((tex_dim,tex_dim,1), dtype=np.float32))

    if with_metallic:
        mesh.material.texture_maps['metallic'] = o3d.t.geometry.Image(metallic_start + np.zeros((tex_dim,tex_dim,1), dtype=np.float32))
        
    if transform is None:
        transform = np.eye(4)
    trafo = np.array([
        [-1, 0, 0, center[0]],
        [0, -1, 0, center[1]],
        [0, 0, 1, center[2]],
        [0, 0, 0, 1]])
    mesh.transform(trafo @ transform)
    o3d.t.io.write_triangle_mesh(f'{model_path.split(".")[0]}_1.obj', mesh)
    
    bsdf = mi.load_dict({
                'type': 'principled',
                'base_color': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(mesh.material.texture_maps['albedo'].as_tensor().numpy()),
                    'wrap_mode': 'mirror'
                },
                'roughness': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(mesh.material.texture_maps['roughness'].as_tensor().numpy())
                },
                'metallic': {
                    'type': 'bitmap',
                    'bitmap': mi.Bitmap(mesh.material.texture_maps['metallic'].as_tensor().numpy())
                }
            })
    mesh_mitsuba = mesh.to_mitsuba("mesh", bsdf=bsdf)
    return mesh_mitsuba

def render(in_dir, begin=0, end=200, step=1, mesh_file=None,
            fov_x=45,
            img_size=[640, 480],
            plane=None,
            env_file=None,
            background_image=None, remove=False, move_to_surface=False):
    # Load the mesh object
    if isinstance(mesh_file, str):
        mesh_file = [mesh_file]
    meshes = []
    vertex_nums = []
    remove_indices = []
    for file in mesh_file:
        if os.path.exists(f'{file.split(".")[0]}_albedo.png'):
            tex_map = f'{file.split(".")[0]}_albedo.png'
        else:
            tex_map = f'{file.split(".")[0]}.png'
        if os.path.exists(f'{file.split(".")[0]}_roughness.png'):
            image = Image.open(f'{file.split(".")[0]}_roughness.png').convert('RGB')
            roughness = np.mean(np.array(image))
        else:
            roughness = 1
        if os.path.exists(f'{file.split(".")[0]}_metallic.png'):
            image = Image.open(f'{file.split(".")[0]}_metallic.png').convert('RGB')
            metallic = np.mean(np.array(image))
        else:
            metallic = 0
        mesh_mitsuba = load_input_model(file, tex_map=tex_map, with_roughness=True, with_metallic=True, starting_values=(0, 0.9, 0.5))
        mesh_params = mi.traverse(mesh_mitsuba)
        vertices = np.array(mesh_params["vertex_positions"]).reshape(-1, 3)
        vertex_num = len(vertices)
        meshes.append(mesh_mitsuba)
        vertex_nums.append(vertex_num)
        if remove:
            kd_tree = KDTree(vertices)
            remove_count = 0
            remove_indices_single = []
            for i in range(vertex_num):
                distances, indices = kd_tree.query(vertices[i], k=200)
                if max(distances) > 0.20:
                    remove_count += 1
                    remove_indices_single.append(i)
            print("remove", remove_count,"points")
            remove_indices.append(remove_indices_single)
    print("vertex_nums", vertex_nums)

    # Create the scene
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'prb',
            'hide_emitters': True,
        },
        'sensor': {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'film': {
                'type': 'hdrfilm',
                'width': int(img_size[0]),
                'height': int(img_size[1]),
                'pixel_format': 'rgba',
            },
            'sampler': {
                'type': 'multijitter',
                'sample_count': 64
            },
            'to_world': mi.ScalarTransform4f.look_at(
                origin=center,
                target=[center[0], center[1], center[2]+1],
                up=[0, 1, 0]
            )
        },
        'emitter': {
            'type': 'envmap',
            'filename': env_file,
            'scale': 1.0,
        }
    }

    # Create a white plane for shadows
    if plane is not None:
        a, b, c, d = plane
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        distance = -d / np.linalg.norm([a, b, c])
        translation = normal * distance
        translation += np.array(center)
        translation[0:2] = SIZE - translation[0:2] # to meet with mitsuba setting
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, normal)
        rotation_angle = np.arccos(np.dot(z_axis, normal))
        if np.linalg.norm(rotation_axis) < 1e-6:  # If the normal is already aligned with the z-axis
            rotation = mi.ScalarTransform4f.rotate([0, 0, 1], 0)
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation = mi.ScalarTransform4f.rotate(rotation_axis, np.degrees(rotation_angle))
        rotation = mi.ScalarTransform4f.rotate([0, 0, 1], 180) @ rotation # turn it over
        rotation = mi.ScalarTransform4f.rotate([1, 0, 0], 180) @ rotation # to meet with mitsuba setting
        to_world = mi.ScalarTransform4f.translate(translation) @ rotation @ mi.ScalarTransform4f.scale([10, 10, 1.0])
        plane_mesh = mi.load_dict({
            'type': 'rectangle',
            'to_world': to_world,
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': [0.3, 0.3, 0.2]  # White color
                }
            }
        })
        back_dict = scene_dict.copy()
        back_dict['plane'] = plane_mesh
        scene = mi.load_dict(back_dict)
        image_back = mi.render(scene)
        denoiser = mi.OptixDenoiser(input_size=img_size, albedo=False, normals=False, temporal=False)
        image_back = denoiser(image_back)
        mi.util.write_bitmap(f'{output_folder}/background.png', image_back)
        print(f'Background rendered successfully.')
        
        # the original background
        image_origin = cv2.imread(background_image)
        image_origin = image_origin.astype(np.float32)
        cv2.imwrite(f'{output_folder}/origin.png', image_origin)

    for f in range(begin, end, step):
        print(f'Processing frame {f}...')
        output_fn = f'{output_folder}/{f:05d}_0.png'
        frame_dict = scene_dict.copy()
        if os.path.exists(output_fn) and not args.force:
            print('Frame already exists, skipping.')
        else:
            cur_render_input = f'{in_dir}/{f:05d}.npz'
            if not os.path.exists(cur_render_input):
                print(f'Warning: {cur_render_input} not found, skipping.')
                continue

            # Load the particle data
            np_x, np_v, np_color = ParticleIO.read_particles_3d(cur_render_input)
            print(f'Number of input particles: {len(np_x)}')
            print('object center', np.mean(np_x, axis=0))
            if move_to_surface:
                pos = np_x - center
                distances = (a * pos[:, 0] + b * pos[:, 1] + 
                        c * pos[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
                indices = np.where(distances > 0)[0]
                np_x[indices] = np_x[indices] - distances[indices, np.newaxis]*plane[np.newaxis, :3]
                np_x[indices, 1] -= 0.01

                pos = np_x - center
                distances = (a * pos[:, 0] + b * pos[:, 1] + 
                        c * pos[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
                indices = np.where(distances > 0)[0]
                np_x[indices] = np_x[indices] - distances[indices, np.newaxis]*plane[np.newaxis, :3]
            np_x[:, 0:2] = SIZE - np_x[:, 0:2] # to meet with mitsuba setting
            
            # Update the mesh vertex positions and colors
            for i, mesh_mitsuba in enumerate(meshes):
                mesh_params = mi.traverse(mesh_mitsuba)
                previous_num = 0
                for j in range(i):
                    previous_num += vertex_nums[j]
                points = np_x[previous_num:previous_num+vertex_nums[i]]
                if remove:
                    points[remove_indices[i]] += SIZE
                mesh_params["vertex_positions"] = points.flatten()
                # mesh_params["vertex_color"] = np_color.flatten()/255
                mesh_params.update()
                frame_dict[f'mesh{i}'] = mesh_mitsuba

            # render object
            scene = mi.load_dict(frame_dict)
            image_obj = mi.render(scene)
            denoiser = mi.OptixDenoiser(input_size=img_size, albedo=False, normals=False, temporal=False)
            image_obj = denoiser(image_obj)
            mi.util.write_bitmap(output_fn, image_obj)

            if plane is not None:
                # render object & background
                frame_dict['plane'] = plane_mesh
                scene = mi.load_dict(frame_dict)
                image_all = mi.render(scene)
                denoiser = mi.OptixDenoiser(input_size=img_size, albedo=False, normals=False, temporal=False)
                image_all = denoiser(image_all)
                mi.util.write_bitmap(f'{output_folder}/{f:05d}_1.png', image_all)

                # map shade                
                object_alpha = image_obj[..., 3][..., np.newaxis]
                image_no_shade = image_obj[..., :3] * object_alpha + image_back[..., :3] * (1-object_alpha)
                mi.util.write_bitmap(f'{output_folder}/{f:05d}_2.png', image_no_shade)
                shadow_difference = image_no_shade - image_all[..., :3]
                # print(shadow_difference)
                # print(np.min(np.array(shadow_difference)), np.max(np.array(shadow_difference)))
                # cv2.imwrite(f'{output_folder}/{f:05d}_3.png', np.array(shadow_difference)*255)
                mi.util.write_bitmap(f'{output_folder}/{f:05d}_3.png', shadow_difference)
                epsilon = 0.1
                image_no_shade = np.maximum(image_no_shade, epsilon)
                shadow_ratio = shadow_difference / image_no_shade
                # print(np.min(np.array(shadow_ratio)), np.max(np.array(shadow_ratio)))
                shadow_ratio = np.clip(shadow_ratio, 0, 1)

                # place object back
                shadow_ratio = cv2.cvtColor(np.array(shadow_ratio), cv2.COLOR_BGR2RGB)
                image_shadowed = image_origin * (1 - shadow_ratio)
                # image_shadowed = image_origin - shadow_difference * 255
                import time
                time.sleep(1)
                image_all_pil = cv2.imread(f'{output_folder}/{f:05d}_1.png')
                frame = np.asarray(image_all_pil) * object_alpha + image_shadowed * (1-object_alpha)
                # frame = frame.clip(0, 255).astype(np.uint8)
                cv2.imwrite(f'{output_folder}/{f:05d}_4.png', frame)
                    
            print(f'Frame {f} rendered successfully.')

def images_to_video(image_folder, video_name, fps, background_image=None):
    print('generating video')
    images = [img for img in os.listdir(image_folder) if img.startswith('0') and img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    # only_plane = cv2.imread(f'{output_folder}/background.png', cv2.IMREAD_GRAYSCALE).astype(np.int32) #os.path.join(image_folder, 'background.png')
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, fps, (width, height))

    for i in range(0, len(images)-1, 5):
        frame = cv2.imread(os.path.join(image_folder, images[i+4]))
        video.write(frame)
    video.release()
    print(f'Video saved as {video_name}')

if __name__ == '__main__':
    input_path = args.path
    with open(os.path.join(input_path, 'transform.json'), 'r') as json_file:
        data = json.load(json_file)

    fov = np.array(data['fov'])
    plane_model = np.array(data['plane_model'])
    object_names = data['object_names']
    plane_model[3] = plane_model[3] * scale
    raw_image = Image.open(os.path.join(input_path, 'raw_image.jpg'))
    img_size = np.array([raw_image.size[0], raw_image.size[1]])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    mesh_file = []
    for i, name in enumerate(object_names):
        mesh_file.append(os.path.join(input_path, f'meshes/{name}.obj'))
    video_name = 'output_video.mp4'
    background_image = os.path.join(input_path, 'inpaint/inpainted_all.jpg')
    render(args.in_dir, args.begin, args.end, args.step, mesh_file=mesh_file,
            fov_x=fov_x, img_size=img_size, plane=plane_model, env_file=args.env,
            background_image=background_image, remove=True, move_to_surface=True)
    images_to_video(output_folder, video_name, fps=args.fps)
