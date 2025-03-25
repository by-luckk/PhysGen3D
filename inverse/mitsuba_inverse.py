import os
import mitsuba as mi
import drjit as dr
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color

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

def fit_polynomial(albedo_init_R, albedo_final_R, degree=3):
    albedo_init_R = np.clip(albedo_init_R, 0, 1)
    albedo_final_R = np.clip(albedo_final_R, 0, 1)
    # p_coeff = np.polyfit(albedo_init_R, albedo_final_R, degree-1)
    
    # def f(x):
    #     return np.polyval(p_coeff, x)
    p_coeff = np.polyfit(albedo_init_R, albedo_final_R / (albedo_init_R + 0.1), degree-1)
    
    def f(x):
        return x * np.polyval(p_coeff, x)

    return f

def tone_mapping(albedo_init, albedo_final, roughness_final=None, metallic_final=None):
    front_mask = np.any(albedo_init - albedo_final != [0, 0, 0], axis=-1)
    
    if roughness_final is not None:
        roughness = np.mean(roughness_final[front_mask])
    else:
        roughness = None
    
    if metallic_final is not None:
        metallic = np.mean(metallic_final[front_mask])
    else:
        metallic = None

    albedo_init_masked = albedo_init[front_mask]
    albedo_final_masked = albedo_final[front_mask]
    albedo_init_R = albedo_init_masked[:, 0]
    albedo_final_R = albedo_final_masked[:, 0]
    albedo_init_G = albedo_init_masked[:, 1]
    albedo_final_G = albedo_final_masked[:, 1]
    albedo_init_B = albedo_init_masked[:, 2]
    albedo_final_B = albedo_final_masked[:, 2]
    
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # axs[0].scatter(albedo_init_R, albedo_final_R, c='r', label='R channel', s=1)
    # axs[0].set_xlabel('albedo_init (R)')
    # axs[0].set_ylabel('albedo_final (R)')
    # axs[0].set_title('R channel comparison')
    # axs[0].grid(True)
    
    # axs[1].scatter(albedo_init_G, albedo_final_G, c='g', label='G channel', s=1)
    # axs[1].set_xlabel('albedo_init (G)')
    # axs[1].set_ylabel('albedo_final (G)')
    # axs[1].set_title('G channel comparison')
    # axs[1].grid(True)
    
    # axs[2].scatter(albedo_init_B, albedo_final_B, c='b', label='B channel', s=1)
    # axs[2].set_xlabel('albedo_init (B)')
    # axs[2].set_ylabel('albedo_final (B)')
    # axs[2].set_title('B channel comparison')
    # axs[2].grid(True)

    # plt.tight_layout()
    # plt.savefig('inverse/result/tone_mapping.png')
    d = 5
    poly_r = fit_polynomial(albedo_init_R, albedo_final_R, degree=d)
    poly_g = fit_polynomial(albedo_init_G, albedo_final_G, degree=d)
    poly_b = fit_polynomial(albedo_init_B, albedo_final_B, degree=d)

    x_vals = np.linspace(0, 1, 100)
    plt.plot(x_vals, poly_r(x_vals), label='Fitted Polynomial', color='red')
    plt.plot(x_vals, poly_g(x_vals), label='Fitted Polynomial', color='green')
    plt.plot(x_vals, poly_b(x_vals), label='Fitted Polynomial', color='blue')
    plt.savefig('inverse/result/polynomial.png')

    albedo_new_R = poly_r(albedo_init[:, :, 0])
    albedo_new_G = poly_g(albedo_init[:, :, 1])
    albedo_new_B = poly_b(albedo_init[:, :, 2])
    albedo_new = np.stack([albedo_new_R, albedo_new_G, albedo_new_B], axis=-1)
    save_image(albedo_new, 'new_albedo.png', 'inverse/result')
    return roughness, metallic

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
    roughness, metallic = tone_mapping(albedo_init, albedo_img, roughness_img, metallic_img)

    return albedo_img, roughness_img, metallic_img

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

def load_image_as_array(image_path):
    img = Image.open(image_path)
    return np.array(img) / 255.0


def draw():
    albedo_init = load_image_as_array('inverse/result/initial_albedo.png')
    albedo_final = load_image_as_array('inverse/result/predicted_albedo.png')
    roughness_final = load_image_as_array('inverse/result/predicted_roughness.png')
    metallic_final = load_image_as_array('inverse/result/predicted_metallic.png')
    tone_mapping(albedo_init, albedo_final, roughness_final, metallic_final)

if __name__ == '__main__':
    draw()