import os
import mitsuba as mi
import drjit as dr
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from skimage import color
from tqdm import tqdm

mi.set_variant('cuda_ad_rgb')

def load_reference_image(img_file, mask_box, img_size):
    ref_image = mi.Bitmap(str(img_file))
    ref_image = np.asarray(ref_image.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=False))

    mask_box = np.array(mask_box).astype(int)
    enlarged_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.float32)
    crop_height, crop_width = ref_image.shape[:2]
    enlarged_img[mask_box[1]:mask_box[1] + crop_height, mask_box[0]:mask_box[0] + crop_width] = ref_image
    return mi.TensorXf(enlarged_img)

def save_image(img, name, output_dir):
    """Saves a float image array with range [0..1] as 8 bit PNG"""
    # scale to 0-255
    texture = o3d.core.Tensor(img * 255.0).to(o3d.core.Dtype.UInt8)
    texture = o3d.t.geometry.Image(texture)
    o3d.t.io.write_image(os.path.join(output_dir, name), texture)

def load_input_model(model_path, with_metallic, with_roughness, trafo, tex_map=None, tex_dim=2048, starting_values=(0.5, 1, 0)):
    print(f'Loading {model_path}...')
    albedo_start, roughness_start, metallic_start = starting_values 

    # load mesh (with texture)   
    mesh_old = o3d.io.read_triangle_mesh(str(model_path), True)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_old)
    print(mesh.material.texture_maps.keys())
    if 'albedo' in mesh.material.texture_maps:
        tex_dim = len(np.array(mesh.material.texture_maps['albedo']))
        print('albedo_texture', mesh.material.texture_maps['albedo'])

    # mesh properties
    mesh.compute_vertex_normals()
    # mesh.material.set_default_properties()
    mesh.material.material_name = 'defaultLit' # note: ignored by Mitsuba, just used to visualize in Open3D
    # mesh.compute_uvatlas(tex_dim)

    # Load texture maps
    if tex_map is not None:
        albedo_image = np.array(Image.open(tex_map)) / 255.0
        mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(albedo_image)
    else:
        mesh.material.texture_maps['albedo'] = o3d.t.geometry.Image(albedo_start + np.zeros((tex_dim, tex_dim, 3), dtype=np.float32))

    # Start with empty maps
    if with_roughness:
        mesh.material.texture_maps['roughness'] = o3d.t.geometry.Image(roughness_start + np.zeros((tex_dim,tex_dim,1), dtype=np.float32))
    else:
        mesh.material.scalar_properties['roughness'] = roughness_start

    if with_metallic:
        mesh.material.texture_maps['metallic'] = o3d.t.geometry.Image(metallic_start + np.zeros((tex_dim,tex_dim,1), dtype=np.float32))
    else:
        mesh.material.scalar_properties['metallic'] = metallic_start

    # transform
    rotation_matrix = np.array([
        [-1, 0, 0, 0],
        [0, -1,  0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ]) # to meet with mitsuba settings
    mesh.transform(rotation_matrix @ trafo)

    return mesh

def prepare_scene(mesh, fov_x, img_size, env_file=None):
    sensor_dict = {
        'type': 'perspective',
        'fov': fov_x,
        'fov_axis': 'x',
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0], 
            target=[0, 0, 1],  
            up=[0, 1, 0]
        ),
        'film': {
            'type': 'hdrfilm',
            'width': int(img_size[0]),
            'height': int(img_size[1]),
            'rfilter': { 'type': 'gaussian' },
            'sample_border': True,
            'pixel_format': 'rgb'
        },
        'sampler': {
            'type': 'multijitter',
            'sample_count': 64
        }
    }

    scene_dict = {
        "type": "scene",
        "integrator": {
            'type': 'prb',
            'hide_emitters': True,
        },
        "emitter": {
            "type": "envmap",
            'filename': env_file,
            'scale': 1.0
        },
        # "light": {
        #     "type": "envmap",
        #     "bitmap": mi.Bitmap(array=np.ones((64, 128, 3)), pixel_format=mi.Bitmap.PixelFormat.RGB),
        # },
        # 'emitter': {
        #     'type': 'envmap',
        #     'filename': env_file,
        #     'scale': 1.0
        # },
        "sensor": sensor_dict,
        "neus": mesh
    }

    return mi.load_dict(scene_dict)

def fit_polynomial(x_data, y_data, degree=3):
    def basis_functions(x):
        # return (n_samples, degree)
        return np.vstack([x**i * (1 - x)**i for i in range(1, degree + 1)]).T

    X = basis_functions(x_data)
    y_target = y_data - x_data # f(x) = x + r(x)
    coefficients, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)

    def f(x):
        residual = basis_functions(x) @ coefficients
        return x + residual

    return f

def tone_mapping(albedo_init, albedo_final, roughness_final=None, metallic_final=None, save_dir=None, k=50, use_hsv=False):
    front_mask = np.any(albedo_init - albedo_final != [0, 0, 0], axis=-1)
    valid_mask = np.any(albedo_init != [0, 0, 0], axis=-1)
    albedo_final[~front_mask] = [0, 0, 0]
    plt.imsave(os.path.join(save_dir, 'RGB_modify_masked.png'), albedo_final)
    
    if roughness_final is not None:
        roughness = np.mean(roughness_final[front_mask])
    else:
        roughness = None
    
    if metallic_final is not None:
        metallic = np.mean(metallic_final[front_mask])
    else:
        metallic = None

    if use_hsv:
        color_space_init = color.rgb2hsv(albedo_init)
        color_space_final = color.rgb2hsv(albedo_final)
        color_space_name = 'HSV'
    else:
        color_space_init = albedo_init.copy()
        color_space_final = albedo_final.copy()
        color_space_name = 'RGB'
    color_init_masked = color_space_init[front_mask]  # Shape: (N, 3)
    color_final_masked = color_space_final[front_mask]  # Shape: (N, 3)
    # color_delta = color_final_masked - color_init_masked  # Shape: (N, 3)

    # if use_hsv:
    #     hue_diff = color_delta[:, 0]
    #     hue_diff = (hue_diff + 0.5) % 1.0 - 0.5
    #     color_delta[:, 0] = hue_diff

    # Build a KD-Tree for the masked HSV values
    tree = KDTree(color_init_masked)
    modified_color_space = color_space_init.copy()
    all_pixels = np.argwhere(valid_mask)  # Shape: (M, 2)
    iterator = tqdm(all_pixels, desc="Processing Pixels")

    for idx, (i, j) in enumerate(iterator):
        current_color = color_space_init[i, j]
        distances, neighbor_indices = tree.query(current_color, k=k)
        if k == 1:
            neighbor_indices = [neighbor_indices]
        modified_color = np.mean(color_final_masked[neighbor_indices], axis=0) ##

        if use_hsv:
            modified_color[0] = modified_color[0] % 1.0  # Hue is in [0,1]
            modified_color[1:] = np.clip(modified_color[1:], 0, 1)
        else:
            modified_color = np.clip(modified_color, 0, 1)

        modified_color_space[i, j] = modified_color

    if use_hsv:
        rgb_modified = color.hsv2rgb(modified_color_space)
    else:
        rgb_modified = modified_color_space.copy()

    if save_dir is not None:
        plt.imsave(os.path.join(save_dir, 'RGB_modify_masked.png'), modified_color_space * int(front_mask)[:, :, None])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f'hsl_modify_{"HSV" if use_hsv else "RGB"}.png'
        plt.imsave(os.path.join(save_dir, filename), rgb_modified)
        print(f"Picture saved to {os.path.join(save_dir, filename)}")

    return modified_color_space, roughness, metallic

def run_material_optimization(mesh, ref_image, fov_x, img_size, env_file=None, dataset_dir=None, iterations=100, lr=1):
    mesh = mesh.to_mitsuba('neus')
    print("mark")
    scene = prepare_scene(mesh, fov_x, img_size, env_file)
    print("scene prepared")
    
    params = mi.traverse(scene)
    print(params)
    bsdf_key_prefix = 'neus.bsdf.'
    opt = mi.ad.Adam(lr=lr, mask_updates=False)
    opt[bsdf_key_prefix + 'base_color.data'] = params[bsdf_key_prefix + 'base_color.data']
    opt[bsdf_key_prefix + 'roughness.data'] = params[bsdf_key_prefix + 'roughness.data']
    opt[bsdf_key_prefix + 'metallic.data'] = params[bsdf_key_prefix + 'metallic.data']
    params.update(opt)

    if dataset_dir:
        albedo_init = params[bsdf_key_prefix + 'base_color.data'].numpy()
        save_image(albedo_init, 'initial_albedo.png', dataset_dir)

    def mse(image, ref_img):
        return dr.mean(dr.sqr(image - ref_img))

    losses = []
    for i in range(iterations):
        img = mi.render(scene, params, spp=64)

        viz = np.concatenate([img, ref_image, np.abs(img-ref_image)], axis=1)
        mi.util.write_bitmap(os.path.join(dataset_dir, f'iter_{i}.png'), mi.TensorXf(viz))
        # viz = np.power(viz, 1/2.2) # gamma correction, from linear to sRGB
        # viz = (viz * 255).astype(np.uint8)
        # iio.imwrite(f'inverse/result/iter_{i}.png', viz)

        loss = mse(img[:, :, :3], ref_image)
        dr.backward(loss)
        losses.append(loss)

        opt.step()
        opt[bsdf_key_prefix + 'base_color.data'] = dr.clamp(opt[bsdf_key_prefix + 'base_color.data'], 0.0, 1.0)
        opt[bsdf_key_prefix + 'roughness.data'] = dr.clamp(opt[bsdf_key_prefix + 'roughness.data'], 0.0, 1.0)
        opt[bsdf_key_prefix + 'metallic.data'] = dr.clamp(opt[bsdf_key_prefix + 'metallic.data'], 0.0, 1.0)
        params.update(opt)

        print(f'Iteration {i} complete, loss: {float(loss[0])}')

    plt.plot(losses)
    plt.savefig(os.path.join(dataset_dir, 'loss'))

    albedo_img = params[bsdf_key_prefix + 'base_color.data'].numpy()
    roughness_img = params[bsdf_key_prefix + 'roughness.data'].numpy()
    metallic_img = params[bsdf_key_prefix + 'metallic.data'].numpy()
    save_image(albedo_img, 'predicted_albedo.png', dataset_dir)
    save_image(roughness_img, 'predicted_roughness.png', dataset_dir)
    save_image(metallic_img, 'predicted_metallic.png', dataset_dir)
    np.save(os.path.join(dataset_dir, 'predicted_albedo.npy'), albedo_img)
    np.save(os.path.join(dataset_dir, 'predicted_roughness.npy'), roughness_img)
    np.save(os.path.join(dataset_dir, 'predicted_metallic.npy'), metallic_img)

    print('valid pixel', np.sum(np.any(albedo_init != [0, 0, 0], axis=-1)))
    print('changed pixel', np.sum(np.any(albedo_init - albedo_img != [0, 0, 0], axis=-1)))
    albedo_modify, roughness, metallic = tone_mapping(albedo_init, albedo_img, roughness_img, metallic_img)

    return albedo_modify, roughness, metallic

def predict():
    dataset_dir = 'inverse/result'
    os.makedirs(dataset_dir, exist_ok=True)
    mesh_path = 'outputs/basketball_texture/meshes/basketball1.obj'
    ref_image_path = 'outputs/basketball_texture/objects/basketball1_black.jpg'
    env_file = '../DiffusionLight/output/hdr/basketball.exr'
    img_size = np.array([547, 640])
    mask_box = [231.369140625, 252.01837158203125, 313.2943115234375, 335.44189453125]
    trafo = np.array([[-0.0509056, -0.0865607, 0.0302814, -0.0023638],
            [0.0833808, -0.0292713, 0.0564971, -0.0629785],
            [-0.0381752, 0.051493, 0.0830193, 2.59676],
            [0, 0, 0, 1]])
    tex_map = 'outputs/basketball_texture/meshes/basketball1.png'
    with_metallic = True
    with_roughness = True
    mesh= load_input_model(mesh_path, with_metallic, with_roughness, trafo, tex_map)
    ref_image = load_reference_image(ref_image_path, mask_box, img_size)
    mi.util.write_bitmap('inverse/result/ref.png', ref_image)
    fov = np.array([-0.27350001, -0.32, 0.27250002, 0.31900002])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    
    
    # process ref image
    print("model loaded")
    albedo, roughness, metallic = run_material_optimization(mesh, ref_image, fov_x, img_size, env_file, dataset_dir)

def predict2():
    dataset_dir = 'inverse/result2'
    os.makedirs(dataset_dir, exist_ok=True)
    mesh_path = 'outputs/car_on_road/meshes/car1.obj'
    ref_image_path = 'outputs/car_on_road/objects/car1_black.jpg'
    env_file = '../DiffusionLight/output/hdr/basketball.exr'
    img_size = np.array([2400, 1600])
    mask_box = [946.4363403320312, 881.2467651367188, 1403.7369384765625, 1160.737060546875]
    trafo = np.array([[-0.04149384,  1.66560636,  0.04722319, -0.09910522],
        [ 0.06519273,  0.04882445, -1.66480094,  1.18459051],
        [-1.66499984, -0.03959725, -0.06636181,  6.99658691],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    tex_map = 'outputs/car_on_road/meshes/car1.png'
    with_metallic = True
    with_roughness = True
    mesh= load_input_model(mesh_path, with_metallic, with_roughness, trafo, tex_map)
    ref_image = load_reference_image(ref_image_path, mask_box, img_size)
    mi.util.write_bitmap('inverse/result2/ref.png', ref_image)
    fov = np.array([-1.20000011, -0.80000006,  1.19900007,  0.79900007])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    
    
    # process ref image
    print("model loaded")
    albedo, roughness, metallic = run_material_optimization(mesh, ref_image, fov_x, img_size, env_file, dataset_dir)

def load_image_as_array(image_path):
    img = Image.open(image_path)
    return np.array(img) / 255.0


def draw():
    albedo_init = load_image_as_array('inverse/result/initial_albedo.png')
    albedo_final = load_image_as_array('inverse/result/predicted_albedo.png')
    roughness_final = load_image_as_array('inverse/result/predicted_roughness.png')
    metallic_final = load_image_as_array('inverse/result/predicted_metallic.png')
    tone_mapping(albedo_init, albedo_final, roughness_final, metallic_final, 'inverse/hsv')

if __name__ == '__main__':
    draw()