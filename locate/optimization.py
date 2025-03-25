import cv2
import numpy as np
import drjit as dr
from drjit.cuda import Float, UInt32, TensorXf
import mitsuba as mi
import trimesh
import open3d as o3d
from mitsuba.scalar_rgb import Transform4f as T

SIZE = 10
mi.set_variant('cuda_ad_rgb')

def numpy_to_drjit(np_array):
    if isinstance(np_array, np.ndarray):
        if np_array.dtype == np.float32:
            drjit_array = Float(np_array.flatten())
        elif np_array.dtype == np.float64:
            drjit_array = Float(np_array.flatten())
        elif np_array.dtype == np.int32:
            drjit_array = UInt32(np_array.flatten())
        elif np_array.dtype == np.int64:
            drjit_array = UInt32(np_array.flatten())
    else:
        raise ValueError(f"Unsupported dtype: {np_array.dtype}")
    
    # drjit_array = dr.array(drjit_type, np_array.flatten())
    if np_array.ndim > 1:
        drjit_array = TensorXf(drjit_array, shape=np_array.shape)
    return drjit_array

def apply_transformation(params, opt, initial_vertex_positions):
    opt['trans'] = dr.clamp(opt['trans'], -0.5, 0.5)
    # opt['angle'] = dr.clamp(opt['angle'], -0.5, 0.5)

    trafo = mi.Transform4f.translate([opt['trans'].x, opt['trans'].y, opt['trans'].z])# .rotate([0, 1, 0], opt['angle'] * 100.0)

    params['mesh.vertex_positions'] = dr.ravel(trafo @ initial_vertex_positions)
    params.update()


# def loss_fn(img, ref_img):
#     epsilon = 1e-3
#     mismatch_black = dr.abs(img - ref_img)
#     mismatch_penalty = dr.select(mismatch_black > epsilon, 1.0, 0.0)
#     return dr.sum(mismatch_penalty)

def sigmoid(array):
    # array: m * n * 3
    return 1 / (1 + dr.exp(array))

# def loss_fn(img, ref_img, ref_normal):
#     '''
#     img: m * n * 8, [R, G, B, alpha, depth, normal_x, normal_y, normal_z]
#     ref_img: m * n * 3, black background
#     '''
#     # compute object mask
#     object_mask = np.any(np.asarray(ref_img) != 0, axis=2)

#     # deform mask and change to drjit
#     expanded_mask = np.expand_dims(object_mask, axis=2)
#     expanded_mask = np.repeat(expanded_mask, 3, axis=2)
#     expanded_mask = numpy_to_drjit(expanded_mask.astype(int))

#     # compute loss
#     img_masked = img[:, :, :3] * expanded_mask
#     normal_masked = img[:, :, -3:] * expanded_mask
#     ref_normal_masked = ref_normal * expanded_mask
#     mi.util.write_bitmap('locate/normal_masked.png', normal_masked)
#     mi.util.write_bitmap('locate/ref_normal_masked.png', ref_normal_masked)

#     diff1 = img_masked - numpy_to_drjit(ref_img[..., :3])
#     # diff = sigmoid(img_masked) - sigmoid(ref_img[..., :3])
#     diff1 = dr.sum(dr.abs(diff1))
#     diff2 = normal_masked - numpy_to_drjit(ref_normal_masked)
#     diff2 = dr.sum(dr.abs(diff2))
#     print('\n', diff1, diff2, '\n')
#     return diff1 + diff2

def loss_fn(img, ref_img, ref_normal, ref_depth):
    '''
    img: m * n * 8, [R, G, B, alpha, depth, normal_x, normal_y, normal_z]
    ref_img: m * n * 3, black background
    '''
    # compute object mask
    object_mask = np.any(np.asarray(ref_img) != 0, axis=2)

    # deform mask and change to drjit
    expanded_mask = np.expand_dims(object_mask, axis=2)
    expanded_mask = np.repeat(expanded_mask, 3, axis=2)
    expanded_mask = numpy_to_drjit(expanded_mask.astype(int))

    # compute masked image
    img_masked = img[:, :, :3] * expanded_mask
    normal_masked = img[:, :, -3:] * expanded_mask
    depth_masked = ref_depth * numpy_to_drjit(object_mask.astype(int))
    ref_normal_masked = ref_normal * expanded_mask
    mi.util.write_bitmap('locate/normal_masked.png', normal_masked)
    mi.util.write_bitmap('locate/ref_normal_masked.png', ref_normal_masked)

    # compute loss
    depth = img[:, :, 4]
    diff1 = depth - numpy_to_drjit(depth_masked)
    diff1 = dr.sum(dr.abs(diff1))
    diff2 = normal_masked - numpy_to_drjit(ref_normal_masked)
    diff2 = dr.sum(dr.abs(diff2))
    print('\nloss', diff1, diff2)
    return - (diff1 + diff2)

def optimize(mesh_file, init_metrix, fov_x, img_size, ref_img, mask_box, ref_normal, ref_depth, raw_image=None):
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'aov',
            'aovs': 'dd.y:depth,nn:sh_normal',
            'my_image': {
                'type': 'path',
            }
        },
        'sensor':  {
            'type': 'perspective',
            'fov': fov_x,
            'fov_axis': 'x',
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, 0],
                target=[0, 0, 1],
                up=[0, -1, 0]
            ),
            'film': {
                'type': 'hdrfilm',
                'width': int(img_size[0]),
                'height': int(img_size[1]),
                'rfilter': { 'type': 'gaussian' },
                'sample_border': True,
                'pixel_format': 'rgba'
            },
        },
        'light': {
            'type': 'sphere',  # Use built-in sphere shape
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': [1000, 1000, 1000]  # Bright white light
                }
            },
            'to_world': mi.ScalarTransform4f.look_at(
                origin=[0, 0, -1],  # Slightly ahead of sensor
                target=[0, 0, 0],  # Same direction as sensor
                up=[0, 1, 0]
            ).scale(2000)  # Scale the sphere light source
        },
    }

    # add mesh
    mesh_mitsuba = mi.load_dict({
                'type': 'obj',
                'filename': mesh_file,
                "face_normals": True,
                "bsdf": {
                    "type": "diffuse",
                    "reflectance": {
                        "type": "mesh_attribute",
                        "name": "vertex_color",  # This will be used to visualize our attribute
                    },
                },
                'to_world': mi.ScalarTransform4f(init_metrix),
            })
    attribute_size = mesh_mitsuba.vertex_count() * 3

    # add color
    mesh_mitsuba.add_attribute("vertex_color", 3, [0] * attribute_size)
    mesh_params = mi.traverse(mesh_mitsuba)
    mesh_o3d = o3d.io.read_triangle_mesh(mesh_file)
    color = np.asarray(mesh_o3d.vertex_colors)
    mesh_params["vertex_color"] = color.flatten()
    mesh_params.update()
    scene_dict['mesh'] = mesh_mitsuba
    scene = mi.load_dict(scene_dict)
    image_init = mi.render(scene) # [R, G, B, alpha, depth, normal_x, normal_y, normal_z]
    print(np.max(np.asarray(image_init[:, :, :3])))
    cv2.imwrite('locate/mesh.png', cv2.cvtColor(np.asarray(image_init[:, :, :3]) * 255, cv2.COLOR_RGB2BGR))

    # reference
    ref_img = cv2.imread(ref_img)
    ref_normal = np.load(ref_normal)
    ref_depth = np.load(ref_depth)
    lower_white = np.array([250, 250, 250], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(ref_img, lower_white, upper_white)
    ref_img[white_mask == 255] = [0, 0, 0]
    mask_box = np.array(mask_box).astype(int)
    restored_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    crop_height, crop_width = ref_img.shape[:2]
    restored_img[mask_box[1]:mask_box[1] + crop_height, mask_box[0]:mask_box[0] + crop_width] = ref_img
    cv2.imwrite('locate/restored.png', restored_img)

    # optimization
    params = mi.traverse(scene)
    initial_vertex_positions = dr.unravel(mi.Point3f, params['mesh.vertex_positions'])
    opt = mi.ad.Adam(lr=0.001, mask_updates=True)
    # opt['angle'] = mi.Float(0)
    opt['trans'] = mi.Point3f(0, 0, 0)

    iteration_count = 5
    spp = 16
    loss_hist = []
    for it in range(iteration_count):
        apply_transformation(params, opt, initial_vertex_positions)
        img = mi.render(scene, params, seed=it, spp=spp)
        loss = loss_fn(img, (restored_img / 255), ref_normal, ref_depth)
        dr.backward(loss)
        print('grad', dr.grad(opt['trans']))
        opt.step()
        loss_hist.append(loss)
        print(f"Iteration {it:02d}: error={loss[0]:6f}, trans=[{opt['trans'].x[0]:.4f}, {opt['trans'].y[0]:.4f}, {opt['trans'].z[0]:.4f}]", end='\r')

    image_final = mi.render(scene, spp=1024)
    raw_image = cv2.imread(raw_image)
    image_init = cv2.cvtColor(np.asarray(image_init[:, :, :3]), cv2.COLOR_RGB2BGR)
    image_final = cv2.cvtColor(np.asarray(image_final[:, :, :3]), cv2.COLOR_RGB2BGR)
    cv2.imwrite('locate/initial.png', image_init * 255 * 0.3)
    cv2.imwrite('locate/final.png', image_final * 255 * 0.3)

    trafo = mi.Transform4f.translate([opt['trans'].x, opt['trans'].y, opt['trans'].z])
    final_matrix = np.array(trafo @ init_metrix)
    print(np.array2string(final_matrix, separator=', '))

def demo1():
    mesh_file = 'outputs/basketball_on_court/meshes/basketball1.obj'
    fov = np.array([-0.27350001, -0.32, 0.27250002, 0.31900002])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    img_size = np.array([547, 640])
    ref_img = 'outputs/basketball_on_court/objects/basketball1.jpg'
    ref_normal = 'outputs/basketball_on_court/metric3d/vis/normal.npy'
    ref_depth = 'outputs/basketball_on_court/metric3d/vis/depth.npy'
    mask_box = [231.369140625, 252.01837158203125, 313.2943115234375, 335.44189453125]
    M = np.array([[-0.05090556, -0.08656068,  0.03028138,  0.02371416],
        [ 0.08338085, -0.02927126,  0.05649706, -0.07231506],
        [-0.03817522,  0.05149302,  0.08301933,  2.65207012],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    optimize(mesh_file, M, fov_x, img_size, ref_img, mask_box, ref_normal, ref_depth)

def demo2():
    mesh_file = 'outputs/toy/meshes/toy1.obj'
    fov = np.array([-0.32000001, -0.265    ,   0.31900002 , 0.26400001])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    img_size = np.array([640, 530])
    ref_img = 'outputs/toy/objects/toy1.jpg'
    ref_normal = 'outputs/toy/metric3d/vis/normal.npy'
    mask_box = [130.64154052734375, 184.67767333984375, 511.9110107421875, 389.40081787109375]
    M = np.array([[ 0.        , -0.30901699,  0.95105652,  2.85316955],
    [ 1.        ,  0.        , -0.        ,  0.        ],
    [-0.        ,  0.95105652,  0.30901699,  0.92705098],
    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    optimize(mesh_file, M, fov_x, img_size, ref_img, mask_box, ref_normal)

def demo3():
    mesh_file = 'outputs/dog/meshes/dog1.obj'
    fov = np.array([-0.32000001, -0.265     ,  0.31900002,  0.26400001])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    img_size = np.array([640, 530])
    ref_img = 'outputs/dog/objects/dog1.jpg'
    ref_normal = 'outputs/dog/metric3d/vis/normal.npy'
    ref_depth = 'outputs/dog/metric3d/vis/depth.npy'
    mask_box = [128.88363647460938, 21.873275756835938, 527.708984375, 395.96234130859375]
    # M = np.array([[-0.0027115 , -0.03873944, -0.05001234,  0.05106969],
    #    [ 0.06191624, -0.0118924 ,  0.00585493,  0.03300474],
    #    [ 0.01297525,  0.04865344, -0.03839031,  0.64877544],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    M = np.array([[ 0.16547404,  0.07674885, -0.01267212,  0.0032557 ],
        [-0.02601082,  0.08266435,  0.16100519, -0.03802882],
        [ 0.07331034, -0.14390569,  0.08572849,  0.6725838 ],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    optimize(mesh_file, M, fov_x, img_size, ref_img, mask_box, ref_normal, ref_depth)

def demo4():
    mesh_file = 'outputs/car_on_road/meshes/car1.obj'
    fov = np.array([-1.20000011, -0.80000006,  1.19900007,  0.79900007])
    fov_x = np.arctan(fov[2]) * 180 / np.pi *2
    img_size = np.array([2400, 1600])
    ref_img = 'outputs/car_on_road/objects/car1.jpg'
    ref_normal = 'outputs/car_on_road/metric3d/vis/normal.npy'
    ref_depth = 'outputs/car_on_road/metric3d/vis/depth.npy'
    raw_image = 'data/car_on_road.jpg'
    mask_box = [946.4363403320312, 881.2467651367188, 1403.7369384765625, 1160.737060546875]
    M = np.array([[ 1.52915467e-02,  1.40843849e+00, -1.58181643e-03, -1.62588353e-01],
       [ 3.21420257e-02, -1.93046624e-03, -1.40815422e+00, 1.15845521e+00],
       [-1.40807253e+00,  1.52514532e-02, -3.21610703e-02, 6.69929961e+00],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    optimize(mesh_file, M, fov_x, img_size, ref_img, mask_box, ref_normal, ref_depth, raw_image)

demo4()