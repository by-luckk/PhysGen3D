import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import cv2
import sys
import json
import torch
from scipy.optimize import minimize
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.append(os.getcwd())
from .match_pairs import image_pair_matching
import pytorch3d
from pytorch3d.renderer import (look_at_view_transform, PerspectiveCameras,
                                PointLights, RasterizationSettings, BlendParams,
                                MeshRenderer, MeshRasterizer, SoftPhongShader)
from pytorch3d.io import load_objs_as_meshes

def select_point(pcd, match_points_on_raw, img_size):
    points = np.asarray(pcd.points)
    projection = points / points[:, 2:3]
    projection = projection[:, :2]
    # plt.plot(projection[:, 0], projection[:, 1], 'o')
    # plt.show()

    # bounding box of projection
    min_x, max_x = np.min(projection[:, 0]), np.max(projection[:, 0])
    min_y, max_y = np.min(projection[:, 1]), np.max(projection[:, 1])

    # project to image
    projection[:, 0] = (projection[:, 0] - min_x) / (max_x - min_x) * img_size[1]
    projection[:, 1] = (projection[:, 1] - min_y) / (max_y - min_y) * img_size[0]

    # select points closest to match points
    closest_points = []
    for raw_point in match_points_on_raw:
        distances = np.linalg.norm(projection - raw_point, axis=1)
        closest_index = np.argmin(distances)
        closest_points.append(points[closest_index])

    # show the selected points in 3D
    selected = o3d.geometry.PointCloud()
    selected.points = o3d.utility.Vector3dVector(closest_points)
    # o3d.visualization.draw_geometries([pcd, selected])
    return closest_points

def sample_camera_poses(radius, num_samples, num_up_samples=4, device='cpu'):
    '''
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    '''
    camera_poses = []
    phi = np.linspace(0, np.pi, num_samples)  # Elevation angle
    phi = phi[1:-1]  # Exclude poles
    theta = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angle

    # Generate different up vectors
    up_vectors = [np.array([0, 0, 1])]  # z-axis is up
    for i in range(1, num_up_samples):
        angle = (i / num_up_samples) * np.pi * 2
        up = np.array([np.sin(angle), 0, np.cos(angle)])  # Rotate around y-axis
        up_vectors.append(up)

    for p in phi:
        for t in theta:
            for up in up_vectors:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                position = np.array([x, y, z])[None, :]
                lookat = np.array([0, 0, 0])[None, :]
                up = up[None, :]
                R, T = look_at_view_transform(radius, t, p, False, position, lookat, up)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R
                camera_pose[3, :3] = T
                camera_poses.append(camera_pose)

                # # Attention! Negative! Not normal forward!
                # forward = -(lookat - position) / np.linalg.norm(lookat - position)
                # right = np.cross(up, forward)
                # if np.linalg.norm(right) < 1e-6:  # Check for collinearity
                #     right = np.array([1, 0, 0])
                # right = right / np.linalg.norm(right)
                # up = np.cross(forward, right)

                # camera_pose = np.eye(4)
                # camera_pose[:3, 0] = right
                # camera_pose[:3, 1] = up
                # camera_pose[:3, 2] = forward
                # camera_pose[:3, 3] = position
                # print('old', camera_pose[:3, :3], position)
    print('total poses', len(camera_poses))
    return torch.tensor(np.array(camera_poses), device=device)

def render_image(mesh, camera_poses, width=640, height=480, fov=1, device='cpu'):
    camera_poses = torch.tensor(camera_poses, device=device)
    if len(camera_poses.shape) == 2:
        camera_poses = camera_poses[None, :]
    # Render and save images from different camera poses
    mesh = load_objs_as_meshes([mesh], device=device)
    R = camera_poses[:, :3, :3]
    T = camera_poses[:, 3, :3]
    num_poses = camera_poses.shape[0]
    cameras = PerspectiveCameras(R=R, T=T, device=device,
        focal_length=torch.ones(num_poses, 1) * 0.5 * width / np.tan(fov / 2),  # Calculate focal length from FOV in radians
        principal_point=torch.tensor((width/2, height/2)).repeat(num_poses).reshape(-1, 2),  #different order from image_size!!
        image_size=torch.tensor((height, width)).repeat(num_poses).reshape(-1, 2),
        in_ndc=False)
    # print(cameras.get_world_to_view_transform().get_matrix()) # T is matrix[3, :3]
    light_location = torch.linalg.inv(camera_poses)[:, 3, :3]
    lights = PointLights(location=light_location, device=device)
    raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            blend_params=BlendParams(background_color=(0,0,0)),
            cameras=cameras,
            lights=lights
        )
    )
    fragments = renderer.rasterizer(mesh.extend(num_poses))
    depth = fragments.zbuf.squeeze().cpu().numpy()
    rendered_images = renderer(mesh.extend(num_poses))
    color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)
    # if num_poses>1:
    #     for i in range(num_poses):
    #         plt.imsave(f'locate/render_result/rendered_image_{i}.png', color[i])
    #         plt.imsave(f'locate/render_result/depth_image_{i}.png', depth[i])
    return color, depth

def render_multi_images(mesh, width=640, height=480, fov=1, radius=3.0, num_samples=6, num_ups=2, device='cpu'):
    # Sample camera poses
    camera_poses = sample_camera_poses(radius, num_samples, num_ups, device)

    # Calculate intrinsics
    # aspect_ratio = width / height # modified
    fx = 0.5 * width / np.tan(fov / 2)
    fy = fx # * aspect_ratio
    cx, cy = width / 2, height / 2
    camera_intrinsics = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    color, depth = render_image(mesh, camera_poses, width, height, fov, device)
    return color, depth, camera_poses, camera_intrinsics

def project_2d_to_3d(image_points, depth, camera_intrinsics, camera_pose):
    """
    Project 2D image points to 3D space using the depth map, camera intrinsics, and pose.

    :param image_points: Nx2 array of image points
    :param depth: Depth map
    :param camera_intrinsics: Camera intrinsic matrix
    :param camera_pose: 4x4 camera pose matrix
    :return: Nx3 array of 3D points in world coordinates
    """
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    # Convert image points to normalized device coordinates (NDC)
    ndc_points = np.zeros((image_points.shape[0], 3))
    for i, (u, v) in enumerate(image_points):
        z = depth[int(v), int(u)]
        x = - (u - cx) * z / fx
        y = - (v - cy) * z / fy
        ndc_points[i] = [x, y, z]
    valid_mask = ndc_points[:, 2] > 0
    ndc_points = ndc_points[valid_mask]
    # ndc_points = np.vstack((ndc_points, np.zeros(3), [[0, 0, 0]])) # modified
    # Convert from camera coordinates to world coordinates
    ndc_points_homogeneous = np.hstack((ndc_points, np.ones((ndc_points.shape[0], 1))))
    world_points_homogeneous = ndc_points_homogeneous @ np.linalg.inv(camera_pose)
    return world_points_homogeneous[:, :3], valid_mask

def plot_mesh_with_points(mesh, points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                    triangles=mesh.faces, alpha=0.5, edgecolor='none', color='lightgrey')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', s=10)
    # ax.scatter([2.853], [0], [0.927], color='green', s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    ax.set_title('3D Mesh with Projected Points')
    # for i in range(10):
    #     angle = 360 / 5 * i
    #     ax.view_init(elev=10., azim=angle)
    #     plt.savefig(filename.split('.')[0] + f'_{i}.png')
    plt.savefig(filename)
    plt.clf()
    # new_mesh = mesh.copy()
    # new_vertices = np.vstack((new_mesh.vertices, points))
    # new_faces = new_mesh.faces.copy()
    # new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    # new_mesh.export(filename.split('.')[0] + '.ply', file_type='ply')

def plot_image_with_points(image, points, save_dir):
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], color='red', s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points on Original Image')
    plt.savefig(save_dir)
    plt.clf()

def fit_plane(pcd, filter_z=1, distance_threshold=0.002):
    points = np.asarray(pcd.points)
    mask = points[:, 2] <= filter_z
    pcd_filtered = pcd.select_by_index(np.where(mask)[0])
    plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=distance_threshold,
                                                       ransac_n=3,
                                                       num_iterations=1000)
    points = np.asarray(pcd.points)
    projection = points / points[:, 2:3]
    projection = projection[:, :2]
    # plt.plot(projection[:, 0], projection[:, 1], 'o')
    # plt.show()

    # bounding box of projection
    min_x, max_x = np.min(projection[:, 0]), np.max(projection[:, 0])
    min_y, max_y = np.min(projection[:, 1]), np.max(projection[:, 1])
    field_of_view = np.array([min_x, min_y, max_x, max_y])
    return plane_model, field_of_view

def estimate_pose(mesh_file, pcd_file, ref_img, raw_img, mask_box, out_dir, num_samples=8, num_ups=1):
    # load 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh = trimesh.load(mesh_file)
    ref_img = cv2.imread(ref_img, cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.imread(raw_img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    pcd = o3d.io.read_point_cloud(pcd_file)
    plane_model, field_of_view = fit_plane(pcd, 1, 0.002)
    fov = np.arctan((field_of_view[2] - field_of_view[0]) / 2) * 2

    # Calculate suitable radius
    bounding_box = mesh.bounds
    center = (bounding_box[0] + bounding_box[1]) / 2
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    fov_radians = fov / 2
    radius = (max_dimension / 2) / np.tan(fov_radians)
    radius = 2 * radius
    print('rendering radius', radius)

    # Render multimle images and feature matching
    print('rendering objects...')
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(mesh_file, 
                                                                          raw_img.shape[1],
                                                                          raw_img.shape[0], fov, radius=radius,
                                                                          num_samples=num_samples, num_ups=num_ups, device=device)
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    print('matching features...')
    best_pose, match_result = image_pair_matching(grays, ref_img, out_dir,
                                    resize=[-1], viz=False, save=False, keypoint_threshold=0.001, match_threshold=0.01)
    chosen_pose = camera_poses[best_pose].cpu().numpy()
    print('best_pose', np.array2string(chosen_pose, separator=', '))
    print('matched point number', np.sum(match_result['matches']>-1))
    plt.imsave(os.path.join(out_dir, 'best_pose_rendering.png'), colors[best_pose])
    # chosen_pose[:, 1:3] = -chosen_pose[:, 1:3] # Change due to pyrender's special coordinates

    # Matched points on mesh
    image_points = match_result['keypoints0'][match_result['matches']>-1]
    world_points, valid_mask = project_2d_to_3d(image_points, depths[best_pose],
                                                camera_intrinsics, chosen_pose)
    image_points = image_points[valid_mask]
    plot_mesh_with_points(mesh, world_points, os.path.join(out_dir, 'points_on_3D.png'))
    plot_image_with_points(depths[best_pose], image_points, os.path.join(out_dir, 'points_on_2D.png'))

    # Matched points on original picture
    match_points_on_mask = match_result['keypoints1'][match_result['matches'][match_result['matches']>-1]]
    match_points_on_mask = match_points_on_mask[valid_mask]
    sclae_x = (mask_box[2]-mask_box[0]) / ref_img.shape[1]
    sclae_y = (mask_box[3]-mask_box[1]) / ref_img.shape[0]
    match_points_on_raw = match_points_on_mask * np.array([sclae_x, sclae_y]) + np.array([mask_box[0], mask_box[1]])
    plot_image_with_points(raw_img, match_points_on_raw, os.path.join(out_dir, 'points_original.png'))

    success, rvec, tvec = cv2.solvePnP(
        np.float32(world_points),
        np.float32(match_points_on_raw),
        np.float32(camera_intrinsics),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    assert success
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    world_2_cam = np.eye(4, dtype=np.float32)
    world_2_cam[:3, :3] = rotation_matrix
    world_2_cam[:3, 3] = tvec.squeeze()
    print('Optimized pose', world_2_cam)

    world_2_cam_render = np.eye(4, dtype=np.float32)
    world_2_cam_render[:3, :3] = np.linalg.inv(rotation_matrix)
    world_2_cam_render[3, :3] = tvec.squeeze() # change due to pytorch3D setting
    world_2_cam_render[:, :2] = -world_2_cam_render[:, :2] # change due to pytorch3D setting
    print('Optimized pose for rendering', np.array2string(world_2_cam_render, separator=', '))
    color, _ = render_image(mesh_file, world_2_cam_render, raw_img.shape[1], raw_img.shape[0], fov, device)
    plt.imsave(os.path.join(out_dir, 'optimized_rendering.png'), color[0])

    # rescaled mesh points
    # mesh points in camera space
    mesh_points = np.hstack((world_points, np.ones((world_points.shape[0], 1)))) @ (world_2_cam).T
    mesh_points = mesh_points[:, :3]
    pcd_points = select_point(pcd, match_points_on_raw, raw_img.shape)

    def objective(scale, mesh_points, pcd_points, plane_model):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        # loss = np.sum(np.linalg.norm(transformed_points - pcd_points, axis=1))
        # a, b, c, d = plane_model
        # distances = (a * transformed_points[:, 0] + b * transformed_points[:, 1] +
        #                   c * transformed_points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        # distances *= np.sign(c)
        # penalty = np.sum(distances[distances > 0] ** 2) * 5
        return loss # + penalty

    initial_scale = 1
    result = minimize(objective, initial_scale, args=(mesh_points, pcd_points, plane_model), method='L-BFGS-B')
    optimal_scale = result.x[0]
    print('Rescale:', optimal_scale)

    # combine pose and transformation
    S = np.array([[optimal_scale, 0, 0, 0],
                  [0, optimal_scale, 0, 0],
                  [0, 0, optimal_scale, 0],
                  [0, 0, 0, 1]])
    M = np.dot(S, world_2_cam)
    print('final matrix')
    print(np.array2string(M, separator=', '))
    print('plane model')
    print(np.array2string(plane_model, separator=', '))
    print('field of view')
    print(np.array2string(field_of_view, separator=', '))
    return M, plane_model, field_of_view

def demo1():
    # from outputs/chair/mask/mask.json
    mask_box = [22.304840087890625, 77.94026184082031, 309.89825439453125, 329.07733154296875]
    M, plane_model, fov = estimate_pose('outputs/chair/meshes/chair1.obj',
                    'outputs/chair/metric3d/pcd/raw_image.ply',
                    'outputs/chair/objects/chair1.jpg',
                    'outputs/chair/objects/chair.jpg',
                    mask_box, 'outputs/chair')

def demo2():
    # outputs/basketball_on_court/mask/mask.json
    mask_box = [231.369140625, 252.01837158203125, 313.2943115234375, 335.44189453125]
    M, plane_model, fov = estimate_pose('outputs/basketball_texture/meshes/basketball1.obj',
                    'outputs/basketball_texture/metric3d/pcd/raw_image.ply',
                    'outputs/basketball_texture/objects/basketball1.jpg',
                    'outputs/basketball_texture/raw_image.jpg',
                    mask_box, 'outputs/basketball_texture/objects')

def demo3():
    # outputs/basketball_on_court/mask/mask.json
    mask_box = [128.88363647460938, 21.873275756835938, 527.708984375, 395.96234130859375]
    M, plane_model, fov = estimate_pose('outputs/dog/meshes/dog1.obj',
                    'outputs/dog/metric3d/pcd/raw_image.ply',
                    'outputs/dog/objects/dog1.jpg',
                    'outputs/dog/raw_image.jpg',
                    mask_box, 'outputs/dog/objects')

def demo4():
    mask_box = [946.4363403320312, 881.2467651367188, 1403.7369384765625, 1160.737060546875]
    M, plane_model, fov = estimate_pose('outputs/car_on_road/meshes/car1.obj',
                    'outputs/car_on_road/metric3d/pcd/raw_image.ply',
                    'outputs/car_on_road/objects/car1_black.jpg',
                    'outputs/car_on_road/raw_image.jpg',
                    mask_box, 'outputs/car_on_road/objects')
    
def demo5():
    mask_box = [1833.97900390625, 1403.532470703125, 3425.2041015625, 2250.368408203125]
    M, plane_model, fov = estimate_pose('outputs/car_in_mountain/meshes/car1.obj',
                    'outputs/car_in_mountain/metric3d/pcd/raw_image.ply',
                    'outputs/car_in_mountain/objects/car1_black.jpg',
                    'outputs/car_in_mountain/raw_image.jpg',
                    mask_box, 'outputs/car_in_mountain/objects')

def demo6():
    mask_box = [67.09408569335938, 117.60181427001953, 308.1783447265625, 362.48944091796875]
    M, plane_model, fov = estimate_pose('outputs/apple2/meshes/apple1.obj',
                    'outputs/apple2/depth/pcd/raw_image.ply',
                    'outputs/apple2/objects/apple1_black.jpg',
                    'outputs/apple2/raw_image.jpg',
                    mask_box, 'locate/temp', num_samples=8)

def demo(in_path, out_path, name):
    with open(in_path + '/mask/mask.json') as f:
        mask = json.load(f)
    mask_box = mask[1]['box']
    M, plane_model, fov = estimate_pose(in_path + f'/meshes/{name}.obj',
                    in_path + '/depth/pcd/raw_image.ply',
                    in_path + f'/objects/{name}_black.jpg',
                    in_path + '/raw_image.jpg',
                    mask_box, out_path, num_samples=8, num_ups=1)
                    
if __name__ == "__main__":
    # pcd = o3d.io.read_point_cloud('outputs/car_in_mountain/depth/pcd/raw_image.ply')
    # plane_model, fov = fit_plane(pcd, distance_threshold=0.01)
    # print(np.array2string(np.asarray(plane_model), separator=', '))
    # demo('outputs/car_in_mountain2', 'locate/car_in_mountain', 'car')
    # demo('outputs/greenapple', 'locate/greenapple', 'apple')
    demo('outputs/DSC00542', 'outputs/DSC00542/objects/orange1', 'orange1')
    