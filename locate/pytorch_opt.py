import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import json
import torch.nn.functional as F
import open3d as o3d
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams, PointLights
    )
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Transform3d, axis_angle_to_matrix, matrix_to_axis_angle
from torch.optim import SGD, lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pickle

def silhouette_loss_func(pred_masks, gt_masks):
    _diff = (pred_masks - gt_masks) ** 2
    return (torch.sum(_diff, dim=(1, 2)) / torch.sum(gt_masks, dim=(1, 2))).view(-1)

def dice_loss_func(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def normal_loss_fun(normal_map, gt_map):
    # Normalize the normal maps while avoiding division by zero
    normal_map_normalized = normal_map / (normal_map.norm(dim=2, keepdim=True) + 1e-8)
    gt_map_normalized = gt_map / (gt_map.norm(dim=2, keepdim=True) + 1e-8)

    # Create mask to check if either normal_map or gt_map is all zeros at any pixel
    normal_map_zero_mask = (normal_map.norm(dim=2) == 0) 
    gt_map_zero_mask = (gt_map.norm(dim=2) == 0)
    # Calculate cosine similarity
    cosine_similarity = (normal_map_normalized * gt_map_normalized).sum(dim=2)
    cosine_similarity[normal_map_zero_mask] = 0
    cosine_similarity[gt_map_zero_mask] = 1
    normal_loss = 1 - cosine_similarity
    non_zero_count = (~gt_map_zero_mask).sum().item()

    # # paint normal_loss
    # normal_loss_map = ((1-cosine_similarity)/2).squeeze().detach().cpu().numpy()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(normal_loss_map, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    # plt.colorbar(label='Normal Loss Value')
    # plt.title('Normal Loss Map')
    # plt.savefig('locate/NormalLossMap.png')
    return normal_loss.sum() / non_zero_count

def depth_loss_fun(depth_map, gt_map):
    depth_map_zero_mask = (depth_map <= 0)
    gt_map_zero_mask = (gt_map <= 0)
    depth_loss = torch.abs(depth_map - gt_map)
    depth_loss[depth_map_zero_mask] = 1
    depth_loss[gt_map_zero_mask] = 0
    non_zero_count = (~gt_map_zero_mask).sum().item()
    
    # # paint depth_loss
    # depth_loss_map = depth_loss.squeeze().detach().cpu().numpy()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(depth_loss_map, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Depth Loss Value')
    # plt.title('Depth Loss Map')
    # plt.savefig('locate/DepthLossMap.png')
    # plt.close()
    return depth_loss.sum() / non_zero_count

def visualize_results(ref_image, rendered_image, title, save_dir):
    if title=="Initial Depth":
        max_depth = np.max(rendered_image)
        ref_image /= max_depth
        rendered_image /= max_depth
        ref_image = np.clip(ref_image, 0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(ref_image.squeeze())
    ax[0].set_title("Reference")
    ax[0].axis("off")
    ax[1].imshow(rendered_image.squeeze())
    ax[1].set_title(title)
    ax[1].axis("off")
    plt.savefig(os.path.join(save_dir, f'{title}'))
    plt.close()

def phong_normal_shading(meshes, fragments) -> torch.Tensor:
    faces = meshes.faces_packed()  # (F, 3)
    vertex_normals = meshes.verts_normals_packed()  # (V, 3)
    faces_normals = vertex_normals[faces]
    ones = torch.ones_like(fragments.bary_coords)
    pixel_normals = pytorch3d.ops.interpolate_face_attributes(
        fragments.pix_to_face, ones, faces_normals
    )
    return pixel_normals

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def compute_transform_matrix(rotation, transition, scale, device):
    R = rotation_6d_to_matrix(rotation.unsqueeze(0))
    S = torch.diag_embed(torch.stack([scale, scale, scale])).unsqueeze(0)
    RS = R @ S
    T = transition.view(1, 3).to(device)
    transform_matrix = torch.eye(4, device=device).repeat(1, 1, 1)
    transform_matrix[:, :3, :3] = RS
    transform_matrix[:, :3, 3] = T
    return transform_matrix

def apply_transform(rotation, transition, scale, verts, device):
    # calculate transform matrix
    transform_matrix = compute_transform_matrix(rotation, transition, scale, device)
    # apply transformation
    V = verts.shape[0]
    verts_homo = torch.cat([verts, torch.ones((V, 1), device=device)], dim=1)
    verts_transformed_homo = torch.matmul(verts_homo, transform_matrix[0].T)
    verts_transformed = verts_transformed_homo[:, :3]
    return verts_transformed

def render_depth(ply_file, fov_x, img_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = o3d.io.read_triangle_mesh(ply_file)
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
    faces = torch.tensor(mesh.triangles, dtype=torch.int64).to(device)
    mesh = Meshes(verts=[vertices], faces=[faces])
    R, T = torch.eye(3)[None, ...], torch.zeros(1, 3)
    R[:2, :2] = -R[:2, :2]
    cameras = PerspectiveCameras(
        device=device,
        R=R.to(device),
        T=T.to(device),
        focal_length=((0.5 * img_size[1]) / np.tan(fov_x / 2),),  # Calculate focal length from FOV in radians
        principal_point=((img_size[1] / 2, img_size[0] / 2),),
        image_size=((img_size[0], img_size[1]),),
        in_ndc=False
    )
    raster_settings = RasterizationSettings(
        image_size=(int(img_size[0]), int(img_size[1])),
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    blend_params = BlendParams(background_color=(0, 0, 0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )

    fragments = renderer.rasterizer(mesh)
    depth = fragments.zbuf.squeeze().cpu().numpy()
    return depth

    
def optimize_mesh_pose(mesh_file, ref_depth, mask, fov_x, img_size, ref_normal=None,
                       pre_transform_matrix=None, num_iterations=100, lr=0.01, save_dir=None):
    '''
    init_transform: [axis_angle, transition] each represented as a 1*3 tensor
    '''
    # Load mesh and texture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = load_objs_as_meshes([mesh_file], device=device)
    original_verts = mesh.verts_packed().clone()

    # Initialize params
    pre_matrix_tensor = torch.tensor(pre_transform_matrix, dtype=torch.float32, device=device)
    M = pre_matrix_tensor[:3, :3]
    U, S, Vh = torch.linalg.svd(M)
    rotation = U @ Vh
    # axis_angle = torch.nn.Parameter(matrix_to_axis_angle(rotation), requires_grad=True)
    axis_angle = torch.flatten(rotation[:2, :3])
    axis_angle = torch.nn.Parameter(axis_angle, requires_grad=True)
    scale = torch.nn.Parameter(torch.mean(S), requires_grad=True)
    transition = torch.nn.Parameter(pre_matrix_tensor[:3, 3], requires_grad=True)
    print('initial', axis_angle, scale, transition)

    # Create a camera with initial pose and specified FOV
    R = torch.eye(3).unsqueeze(0).to(device)
    R[:, :2] = -R[:, :2]
    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=torch.zeros(1, 3).to(device),
        focal_length=((0.5 * img_size[1]) / np.tan(fov_x / 2),),  # Calculate focal length from FOV in radians
        principal_point=((img_size[1] / 2, img_size[0] / 2),),
        image_size=((img_size[0], img_size[1]),),
        in_ndc=False
    )

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(int(img_size[0]), int(img_size[1])),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader = SoftPhongShader(device=device, cameras=cameras)
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    shader_silhouette = SoftSilhouetteShader(blend_params=blend_params)
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader_silhouette
    )

    # Target silhouette and normal maps
    masked_depth = np.zeros_like(ref_depth)
    masked_depth[mask] = ref_depth[mask]
    mask_tensor = torch.tensor(mask, device=device, dtype=torch.float32)
    depth_tensor = torch.tensor(masked_depth, device=device)
    if ref_normal is not None:
        masked_normal = np.zeros_like(ref_normal)
        masked_normal[mask] = ref_normal[mask]
        normal_tensor = torch.tensor(masked_normal, device=device)

    # Render initial result
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        verts_transformed = apply_transform(axis_angle, transition, scale, original_verts, device)
        initial_mesh = mesh.update_padded(verts_transformed.unsqueeze(0))
        fragments = rasterizer(initial_mesh)
        initial_depth = fragments.zbuf.detach().cpu().numpy()
        visualize_results(masked_depth, initial_depth, "Initial Depth", save_dir)

        if ref_normal is not None:
            initial_normal = phong_normal_shading(initial_mesh, fragments).detach().cpu().numpy()
            # print('initial_normal', initial_normal.shape) # b * h * w * faces_per_pixel * 3
            initial_normal = np.mean(initial_normal, axis=-2)
            visualize_results(masked_normal, initial_normal, "Initial Normal", save_dir)

        initial_image = renderer(initial_mesh).detach().cpu().numpy()
        visualize_results(mask, initial_image[..., 3], "Initial Silhouette", save_dir)

    # Optimizer
    optimizer = SGD([axis_angle, scale, transition], lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    # Optimization loop
    loss_list = []
    para_list = []
    for i in range(num_iterations):
        optimizer.zero_grad()
        # Apply transformation
        verts_transformed = apply_transform(axis_angle, transition, scale, original_verts, device)
        transformed_mesh = mesh.update_padded(verts_transformed.unsqueeze(0))
        # Render normal map and compute normal loss
        fragments = rasterizer(transformed_mesh)
        # normal_loss = 0
        # if ref_normal is not None:
        #     normal_map = phong_normal_shading(transformed_mesh, fragments).squeeze() # it's time consuming
        #     normal_loss = normal_loss_fun(normal_map, normal_tensor)
        # Render depth map and compute depth loss
        depth_map = fragments.zbuf.squeeze()
        depth_loss = depth_loss_fun(depth_map, depth_tensor)
        # Render silhouette
        silhouette = shader_silhouette(fragments, transformed_mesh)[..., 3].squeeze()
        # silhouette_loss = torch.nn.functional.mse_loss(silhouette, mask_tensor)
        silhouette_loss = silhouette_loss_func(silhouette.unsqueeze(0), mask_tensor.unsqueeze(0)) / 2
        dice_loss = dice_loss_func(silhouette.unsqueeze(0), mask_tensor.unsqueeze(0)) / 2
        # Total loss (weighted sum of silhouette and normal loss)
        loss = silhouette_loss + dice_loss + depth_loss #+ normal_loss
        loss_list.append([silhouette_loss.item(), dice_loss.item(), depth_loss.item()])#, normal_loss.item(), depth_loss.item()])
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(f"Iteration {i + 1}/{num_iterations}, Loss: {silhouette_loss.item()}, {dice_loss.item()}, {depth_loss.item()}") # {normal_loss.item()}, {depth_loss.item()}, \n",
            # f"transform: {axis_angle}, {transition}, {scale}", scale.grad)
        para_list.append((axis_angle, transition, scale))
        if (i + 1) % 20 == 0 and (save_dir is not None):
            # Render final result
            final_render = renderer(transformed_mesh).detach().cpu().numpy()
            visualize_results(mask, final_render[..., 3], f"Iter {i+1} Silhouette", save_dir)
    
    # save loss curve
    if save_dir is not None:
        with open(os.path.join(save_dir, 'parameter_list.pkl'), 'wb') as f:
            pickle.dump(para_list, f)
        loss_list = torch.tensor(loss_list).cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.plot(loss_list[:, 0], label='Silhouette Loss')
        plt.plot(loss_list[:, 1], label='Dice Loss')
        plt.plot(loss_list[:, 2], label='Depth Loss')
        # plt.plot(loss_list[:, 3], label='Depth Loss')

        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss.png'))
    
    transform_matrix = compute_transform_matrix(axis_angle, transition, scale, device='cpu').detach().numpy()[0]
    print('Before optimizing:', np.array2string(pre_transform_matrix, separator=', '))
    print('After optimizing:', np.array2string(transform_matrix, separator=', '))
    return transform_matrix

def get_white_mask_area(mask_img):
    mask_array = np.array(mask_img)
    white_threshold = 250
    white_mask = mask_array >= white_threshold
    return white_mask

def demo0(folder, name):
    save_dir='locate/pics_dust3r'
    os.makedirs(save_dir, exist_ok=True)
    mesh_file = os.path.join(folder, 'meshes', name + '.obj')
    mask_path = os.path.join(folder, 'mask', 'mask_0.jpg')
    ply_file = os.path.join(folder, 'depth', 'pcd', 'raw_image.ply')
    raw_image = Image.open(os.path.join(folder, 'raw_image.jpg'))
    img_size = np.array([raw_image.size[1], raw_image.size[0]])
    with open(os.path.join(folder, 'transform.json'), 'r') as json_file:
        data = json.load(json_file)
    fov = np.array(data['fov'])
    fov_x = np.arctan(fov[2]) * 2
    pre_transform_matrix = np.array(data['matrices'])[0]
    ref_depth = render_depth(ply_file, fov_x, img_size)
    np.save(os.path.join(save_dir, 'ref_depth.npy'), ref_depth)
    # mask reference
    mask_img = Image.open(mask_path).convert('L')
    resized_mask = mask_img.resize((img_size[1], img_size[0]), Image.LANCZOS)
    white_mask = get_white_mask_area(resized_mask)
    optimized_pose = optimize_mesh_pose(
        mesh_file=mesh_file,
        ref_depth=ref_depth,
        mask=white_mask,
        pre_transform_matrix=pre_transform_matrix,
        fov_x=fov_x,
        img_size=img_size,
        num_iterations=300,
        lr=0.001,
        save_dir=save_dir
    )
    print("Optimized Pose:", optimized_pose)


# Example usage
def demo1():
    mesh_file = 'outputs/car_on_road/meshes/car1.obj'
    mask_path = 'outputs/car_on_road/mask/mask_0.jpg'
    ref_depth = np.load('outputs/car_on_road/metric3d/vis/depth.npy')
    ref_normal = np.load('outputs/car_on_road/metric3d/vis/normal.npy')
    img_size = np.array([1600, 2400])
    fov = np.array([-1.20000011, -0.80000006,  1.19900007,  0.79900007])
    fov_x = np.arctan(fov[2]) * 2
    pre_transform_matrix = [[ 1.52915467e-02,  1.40843849e+00, -1.58181643e-03, -1.62588353e-01],
        [ 3.21420257e-02, -1.93046624e-03, -1.40815422e+00, 1.15845521e+00],
        [-1.40807253e+00,  1.52514532e-02, -3.21610703e-02, 6.69929961e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]]
    # mask reference
    mask_img = Image.open(mask_path).convert('L')
    resized_mask = mask_img.resize((img_size[1], img_size[0]), Image.LANCZOS)
    white_mask = get_white_mask_area(resized_mask)

    optimized_pose = optimize_mesh_pose(
        mesh_file=mesh_file,
        ref_depth=ref_depth,
        ref_normal=ref_normal,
        mask=white_mask,
        pre_transform_matrix=pre_transform_matrix,
        fov_x=fov_x,
        img_size=img_size,
        num_iterations=200,
        lr=1,
        save_dir='locate/pics_sgd'
    )
    print("Optimized Pose:", optimized_pose)

    
def demo2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_file = 'outputs/car_in_mountain/meshes/car1.obj'
    mask_path = 'outputs/car_in_mountain/mask/mask_0.jpg'
    ref_depth = np.load('outputs/car_in_mountain/metric3d/vis/depth.npy')
    ref_normal = np.load('outputs/car_in_mountain/metric3d/vis/normal.npy')
    img_size = np.array([3306, 4960])
    fov = np.array([-1.20000011, -0.80000006,  1.19900007,  0.79900007])
    fov_x = np.arctan(1.2) * 2
    pre_transform_matrix = [[ 0.        , -0.22252093,  0.97492791,  2.92478374],
        [ 1.        ,  0.        , -0.        ,  0.        ],
        [-0.        ,  0.97492791,  0.22252093,  0.6675628 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]]

    # mask reference
    mask_img = Image.open(mask_path).convert('L')
    resized_mask = mask_img.resize((img_size[1], img_size[0]), Image.LANCZOS)
    white_mask = get_white_mask_area(resized_mask)

    optimized_pose = optimize_mesh_pose(
        mesh_file=mesh_file,
        ref_depth=ref_depth,
        ref_normal=ref_normal,
        mask=white_mask,
        init_transform=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        fov_x=fov_x,
        img_size=img_size,
        pre_transform_matrix=pre_transform_matrix,
        num_iterations=100,
        lr=0.003
    )
    print("Optimized Pose:", optimized_pose)

def demo3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_file = 'outputs/potato/meshes/toy1.obj'
    mask_path = 'outputs/potato/mask/mask_0.jpg'
    ref_depth = np.load('outputs/potato/metric3d/vis/depth.npy')
    ref_normal = np.load('outputs/potato/metric3d/vis/normal.npy')
    img_size = np.array([1000, 1000])
    fov = np.array([-0.5     ,   -0.5    ,     0.49900001 , 0.49900003])
    fov_x = np.arctan(fov[2]) * 2
    pre_transform_matrix = [[-0.03543829,  0.36895023, -0.03296233,  0.00939002],
       [-0.07051368, -0.03922666, -0.36325714,  0.04017499],
       [-0.36364625, -0.02834884,  0.07365048,  1.1790783 ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]

    # mask reference
    mask_img = Image.open(mask_path).convert('L')
    resized_mask = mask_img.resize((img_size[1], img_size[0]), Image.LANCZOS)
    white_mask = get_white_mask_area(resized_mask)

    optimized_pose = optimize_mesh_pose(
        mesh_file=mesh_file,
        ref_depth=ref_depth,
        ref_normal=ref_normal,
        mask=white_mask,
        fov_x=fov_x,
        img_size=img_size,
        pre_transform_matrix=pre_transform_matrix,
        num_iterations=100,
        lr=0.003
    )
    print("Optimized Pose:", optimized_pose)

def demo4():
    img_size = [540, 720]
    object_names=['other1', 'book1', 'coke1']
    output_dir = 'outputs/multi2'
    matrices_optimized = []
    ply_file = os.path.join(output_dir, 'depth', 'pcd', 'raw_image.ply')
    with open(os.path.join(output_dir, 'transform.json'), 'r') as json_file:
        data = json.load(json_file)
    fov = np.array(data['fov'])
    fov_x = np.arctan(fov[2]) * 2
    matrices = np.array(data['matrices'])
    ref_depth = render_depth(ply_file, fov_x, img_size)
    mask_img = Image.open(os.path.join(output_dir, 'mask', 'mask_0.jpg')).convert('L')
    resized_mask = mask_img.resize((img_size[1], img_size[0]), Image.LANCZOS)
    white_mask = get_white_mask_area(resized_mask)
    for name in object_names:
        optimized_pose = optimize_mesh_pose(
            mesh_file=os.path.join(output_dir, 'meshes', f'{name}.obj'),
            ref_depth=ref_depth,
            mask=white_mask,
            pre_transform_matrix=matrices[0],
            fov_x=fov_x,
            img_size=img_size,
            num_iterations=300,
            lr=0.001,
            save_dir='locate/pics_sgd'
        )
        matrices_optimized.append(optimized_pose.tolist())
    data["matrices_optimized"] = matrices_optimized

if __name__ == "__main__":
    # demo0('outputs/apple2', 'apple1')
    # demo0('outputs/multi', 'toy1')
    demo4()
# apply_transform(torch.tensor([ 1.0832,  0.8953, -1.1694]), torch.tensor([-0.0491,  0.8910,  4.5986]), 1.4085, None, 'cuda:0')

# [array([[-0.03543829,  0.36895023, -0.03296233,  0.00939002],
#        [-0.07051368, -0.03922666, -0.36325714,  0.04017499],
#        [-0.36364625, -0.02834884,  0.07365048,  1.1790783 ],
#        [ 0.        ,  0.        ,  0.        ,  1.        ]])] [ 0.05529245  0.16883475  0.98409226 -1.1765242 ] [-0.5        -0.5         0.49900001  0.49900003]