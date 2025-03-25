import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import glob
import json
import taichi as ti
import time
import numpy as np
import trimesh
import open3d as o3d
from datetime import datetime
import sys
import argparse
import yaml
from pathlib import Path

def run_mpm_simulation(input_path, velocities, filling=True, pull_to_ground=True,
                    save_video=False, with_gui=False, with_ggui=False, damping=0, friction=0.5,
                    R=512, size=50, scale=1, max_num_particles=2**20, E_default=1e6, density_default=1e3):
    # Setup
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sys.path.append(project_dir)
    from engine.mpm_solver import MPMSolver
    from engine.mesh_io import load_mesh_obj, load_point_cloud_ply, sample_volume_mesh

    # Constants and Configuration
    center = np.array([size / 2, size / 2, size/2])
    ti.init(arch=ti.gpu, device_memory_GB=20, debug=False)
    # ti.init(arch=ti.cuda, device_memory_GB=60, debug=False)

    material_type_colors = np.array([
        [0.1, 0.1, 1.0, 0.8],
        [236.0 / 255.0, 84.0 / 255.0, 59.0 / 255.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0]
    ])

    # Helper function to create output directory
    def create_output_folder(prefix):
        folder = prefix + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(folder)
        return folder

    # Initialize GUI if enabled
    if with_gui:
        gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41, show_gui=False)

    # Initialize GGUI if enabled
    if with_ggui:
        res = (1920, 1080)
        window = ti.ui.Window("Real MPM 3D", res, vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.make_camera()
        camera.position(5, 5, 5)
        camera.lookat(5, 5, 10)
        camera.up(0, -1, 0)
        camera.fov(100)
        particles_radius = 0.02

    # Kernel to set color
    @ti.kernel
    def set_color(ti_color: ti.template(), material_color: ti.types.ndarray(), ti_material: ti.template()):
        for I in ti.grouped(ti_material):
            material_id = ti_material[I]
            color_4d = ti.Vector([0.0, 0.0, 0.0, 1.0])
            for d in ti.static(range(3)):
                color_4d[d] = material_color[material_id, d]
            ti_color[I] = color_4d

    # Render function for GGUI
    def render():
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0, 0, 0))
        set_color(mpm.color_with_alpha, material_type_colors, mpm.material)

        scene.particles(mpm.x, per_vertex_color=mpm.color_with_alpha, radius=particles_radius)

        scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

        canvas.scene(scene)

    # Visualize function for GUI
    def visualize(particles):
        np_x = particles['position'] / size
        screen_x = np_x[:, 0]
        screen_y = (1 - np_x[:, 2])
        screen_pos = np.stack([screen_x, screen_y], axis=-1)
        # screen_pos = screen_pos * 3 -1
        gui.circles(screen_pos, radius=1.5, color=particles['color'])
        if save_video:
            img = gui.get_image()  # get the current frame as an image
            img = np.array(img)  # convert to numpy array
            img = (img * 255).astype(np.uint8)  # convert to 8-bit
            img = img[:, :, :3]  # remove alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gui.clear()
            return img
        else:
            gui.clear()
            return None

    # Convert color to decimal format
    def convert_colors(colors):
        dec_colors = (colors * 255).astype(np.uint8)
        return dec_colors[:, 0]*65536 + dec_colors[:, 1]*256 + dec_colors[:, 2]

    # Create output folder if writing to disk
    output_dir = create_output_folder('./sim_result/sim_result')
    print(output_dir)

    # Video writer setup
    if save_video:
        video_files = glob.glob(os.path.join('sim_result', '*.mp4'))
        next_index = len(video_files) + 1
        video_output = os.path.join('sim_result', f'{next_index}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 24  # frames per second
        video_writer = cv2.VideoWriter(video_output, fourcc, fps, (512, 512))

    # Initialize MPM Solver
    mpm = MPMSolver(res=(R, R, R), quant=True, size=size, unbounded=False, dt_scale=1, E_scale=1,
                    use_diff_E=True, use_adaptive_dt=True, support_plasticity=True, 
                    use_ggui=with_ggui, max_num_particles=max_num_particles, drag_damping=damping)

    # Load transformation and plane model from JSON
    with open(os.path.join(input_path, 'transform.json'), 'r') as json_file:
        data = json.load(json_file)
    # if "matrices_optimized" in data:
    #     matrices = np.array(data['matrices_optimized'])
    # else:
    matrices = np.array(data['matrices'])
    plane = np.array(data['plane_model'])
    object_names = data['object_names']
    if 'E' in data:
        elastics = data['E']
    else:
        elastics = np.ones(len(object_names)) * E_default
    if 'density' in data:
        density = data['density']
    else:
        density = np.ones(len(object_names)) * density_default
    if velocities.ndim == 1:
        velocities = np.tile(velocities, (len(object_names), 1, 1))
        
    plane[3] = plane[3]
    plane *= np.sign(plane[1])
    plain_point = np.array([0, -plane[3]/plane[1], 0]) + center

    # Set gravity and add collider
    g = plane[:3] * 9.8 * scale
    print(f'gravity={g}')
    mpm.set_gravity(g.tolist())
    mpm.add_surface_collider(point=plain_point, normal=-plane[:3],
                             surface=mpm.surface_separate, friction=friction, rebound=0)
    
    # Load object mesh and add particles
    triangles_list = []
    velocity_list = []
    for i, name in enumerate(object_names):
        matrix = matrices[i]
        mesh_file = os.path.join(input_path, f'meshes/{name}.obj')
        reduction_rate = 1
        vertices, triangles, ball_colors = load_mesh_obj(mesh_file, trans_matrix=matrix,
                                                    offset=([0, 0, 0]), reduction_rate=reduction_rate)
        if reduction_rate!=1:
            vertices, indices = vertices
            np.save(os.path.join(output_dir, f'{name}_indices.npy'), indices)
        vertices = vertices + center
        center_3 = np.tile(center, 3)
        triangles = triangles + center_3
        # volume sampling
        # if filling is not None:
        #     filling_file = os.path.join(input_path, f'meshes/{name}_filling_points.npy')
        #     if os.path.exists(filling_file):
        #         filling_points = np.load(filling_file)
        #     else:
        #         print('volume sampling mesh_file')
        #         filling_points = sample_volume_mesh(mesh_file, sample_num=10000, offset=center, trans_matrix=matrix)
        #         np.save(os.path.join(input_path, f'meshes/{name}_filling_points.npy'), filling_points)
        #     # combine filling points and vertices
        #     print('filling_points num', len(filling_points))
        #     vertices = np.concatenate([vertices, filling_points], axis=0)
        #     average_color = np.mean(ball_colors, axis=0)
        #     ball_colors = np.concatenate([ball_colors, np.ones((filling_points.shape[0], 3))], axis=0)

        if pull_to_ground:
            vertices_origin = vertices - center
            a, b, c, d = plane
            distances = (a * vertices_origin[:, 0] + b * vertices_origin[:, 1] + 
                        c * vertices_origin[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
            bias = np.max(distances)
            print('bias', bias)
            print('bias mean', np.mean(distances))
            print('bias min', np.min(distances))
            vertices -= bias*plane[:3]
            displace_3 = np.tile(bias*plane[:3], 3)
            triangles -= displace_3

        down = g / np.linalg.norm(g)
        right = np.array([1, 0, 0])
        forward = np.cross(down, right)
        object_center = np.mean(vertices, axis=0)
        print('object_center', object_center)
        velocity = velocities[i]
        velocity= forward*velocity[1] + down*velocity[2] + right*velocity[0]
        velocity_list.append(velocity)
        mpm.add_particles_color(particles=np.ascontiguousarray(vertices),
                            material=MPMSolver.material_elastic,
                            colors=convert_colors(ball_colors),
                            velocity=velocity,
                            E=elastics[i] * scale,
                            rho=density[i])
        triangles_list.append(triangles)
        particles = mpm.particle_info()
        print('number of points on mesh:', len(particles['position']))

    if filling:
        for i, name in enumerate(object_names):
            if name=="chair1":
                continue
            mpm.add_mesh(triangles_list[i], MPMSolver.material_elastic, sample_density=8, 
                    velocity=velocity_list[i], E=elastics[i] * scale)
        particles = mpm.particle_info()
        print('total point number:', len(particles['position']))

    # Simulation Loop
    start_t = time.time()
    for frame in range(100):
        print(f'frame {frame}')
        t = time.time()

        if with_gui and frame % 1 == 0:
            particles = mpm.particle_info()
            image = visualize(particles)
            if save_video:
                video_writer.write(image)
                cv2.imwrite(f'sim_result/frames/frame_{frame:04d}.png', image)
            if frame % 5 == 0:
                point_cloud = trimesh.points.PointCloud(particles['position'])
                point_cloud.export(f'sim_result/particles/frame_{frame:04d}.ply')
                print('total point number:', len(particles['position']))

        mpm.write_particles(f'{output_dir}/{frame:05d}.npz')
        print(f'Frame total time {time.time() - t:.3f}')
        print(f'Total running time {time.time() - start_t:.3f}')

        if with_ggui:
            render()
            window.show()
        
        mpm.step(1e-2, print_stat=False)

    # Release video writer
    if save_video:
        video_writer.release()
    
    print(output_dir)

def parse_velocities(s):
    """Parse velocities from semicolon-separated vectors"""
    vectors = s.split(';')
    return np.array([list(map(float, v.split(','))) for v in vectors])

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert velocities to numpy array
    if 'velocities' in config:
        config['velocities'] = np.array(config['velocities'])
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Run MPM simulation with config file')
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to configuration file')
    
    parser.add_argument('--size', type=int, help='Override grid size')
    parser.add_argument('--R', type=int, help='Override resolution')
    parser.add_argument('--filling', action='store_true', help='Enable material filling')
    parser.add_argument('--no-pull-to-ground', dest='pull_to_ground', action='store_false',
                       help='Disable pulling to ground')
    parser.add_argument('--E-default', type=float, help='Override Young\'s modulus')
    parser.add_argument('--damping', type=float, help='Override damping coefficient')
    parser.add_argument('--friction', type=float, help='Override friction coefficient')
    parser.add_argument('--input-path', type=str, help='Override input path')
    parser.add_argument('--scale', type=float, help='Override scaling factor')
    parser.add_argument('--velocities', type=parse_velocities, help='Override velocities')
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        raise SystemExit(f"Error: Config file not found at {args.config}")

    override_keys = [
        'size', 'R', 'filling', 'pull_to_ground', 'E_default',
        'damping', 'friction', 'input_path', 'scale', 'velocities'
    ]
    
    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    config['velocities'] = np.array(config['velocities'])
    config['E_default'] = float(config['E_default'])

    run_mpm_simulation(**config)

if __name__ == '__main__':
    main()