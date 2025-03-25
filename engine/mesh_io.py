from plyfile import PlyData
import numpy as np
import open3d as o3d
import trimesh

def load_mesh(fn, scale=1, offset=(0, 0, 0)):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(elements['vertex_indices']):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    return triangles


def rotate_mesh(vertices, angle_x=0, angle_y=0, angle_z=0):
    """Rotate the mesh vertices around the x, y, and z axes by the given angles (in degrees)."""
    radians_x = np.radians(angle_x)
    radians_y = np.radians(angle_y)
    radians_z = np.radians(angle_z)
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(radians_x), -np.sin(radians_x)],
                   [0, np.sin(radians_x), np.cos(radians_x)]])
    
    Ry = np.array([[np.cos(radians_y), 0, np.sin(radians_y)],
                   [0, 1, 0],
                   [-np.sin(radians_y), 0, np.cos(radians_y)]])
    
    Rz = np.array([[np.cos(radians_z), -np.sin(radians_z), 0],
                   [np.sin(radians_z), np.cos(radians_z), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Apply rotation to all vertices
    rotated_vertices = np.dot(vertices, R.T)
    
    return rotated_vertices

def load_mesh_ply(fn, scale, offset, reduction_rate=1, angle_x=0, angle_y=0, angle_z=0):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    vertices = np.vstack((x, y, z)).T
    faces = []
    for face in elements['vertex_indices']:
        if len(face) == 3:
            faces.append(face)
        elif len(face) == 4:
            # Split quad into two triangles
            faces.append([face[0], face[1], face[2]])
            faces.append([face[0], face[2], face[3]])
        else:
            raise ValueError("The mesh contains polygons with more than 4 vertices, which is not supported.")
    faces = np.array(faces)

    if reduction_rate != 1:
        # Create an Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Simplify the mesh
        simplified_mesh = o3d_mesh.simplify_quadric_decimation(int(num_tris * reduction_rate))

        # Extract simplified vertices and faces
        vertices = np.asarray(simplified_mesh.vertices)
        faces = np.asarray(simplified_mesh.triangles)

    # rotate the mesh
    vertices = rotate_mesh(vertices, angle_x, angle_y, angle_z)

    # Scale and offset the simplified vertices
    vertices *= scale
    vertices += offset

    num_tris = len(faces)
    triangles = np.zeros((num_tris, 9), dtype=np.float32)
    
    for i, face in enumerate(faces):
        for d in range(3):
            triangles[i, d * 3 + 0] = vertices[face[d]][0]
            triangles[i, d * 3 + 1] = vertices[face[d]][1]
            triangles[i, d * 3 + 2] = vertices[face[d]][2]

    print('loaded and simplified, face count:', len(faces), 'vertex count:', len(vertices))

    return triangles


def sample_point_mesh(fn, scale, offset, sample_rate=1, angle_x=0, angle_y=0, angle_z=0):
    print(f'loading {fn}')
    vertices = []
    colors = []
    with open(fn, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                color = [float(parts[4]), float(parts[5]), float(parts[6])]
                vertices.append(vertex)
                colors.append(color)
  
    vertices = np.array(vertices)
    colors = np.array(colors)

    # random sample
    if sample_rate != 1:
        sample_indices = np.random.choice(vertices.shape[0], int(vertices.shape[0] * sample_rate), replace=False)
        vertices = vertices[sample_indices]
        colors = colors[sample_indices]
    # rotate the mesh
    vertices = rotate_mesh(vertices, angle_x, angle_y, angle_z)
    vertices = vertices * scale + offset

    return vertices, colors

def load_mesh_obj(fn,  offset, trans_matrix=None, reduction_rate=1, scale=1, angle_x=0, angle_y=0, angle_z=0):
    print(f'loading {fn}')
    mesh = o3d.io.read_triangle_mesh(fn)
    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.triangles)
    if mesh.has_vertex_colors():
        print("Mesh has vertex colors")
        colors = np.asarray(mesh.vertex_colors)
    else:
        print("Mesh does not have vertex colors")
        colors = np.ones((len(mesh.vertices), 3))

    print('loaded and simplified, face count:', len(faces), 'vertex count:', len(vertices))
    
    # rotate the mesh
    if trans_matrix is not None:
        vertices_homo = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
        trans_vertices_homo = vertices_homo.dot(trans_matrix.T)
        vertices = trans_vertices_homo[:, :3]
        vertices += offset
    else:
        vertices = rotate_mesh(vertices, angle_x, angle_y, angle_z)
        vertices = vertices * scale + offset

    num_tris = len(faces)
    triangles = np.zeros((num_tris, 9), dtype=np.float32)
    
    for i, face in enumerate(faces):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = vertices[face[d]][0]
            triangles[i, d * 3 + 1] = vertices[face[d]][1]
            triangles[i, d * 3 + 2] = vertices[face[d]][2]

    # simplify
    if reduction_rate != 1:
        print(f"reduction_rate = {reduction_rate}")
        num_samples = reduction_rate * len(mesh.vertices)
        sample_indices = np.random.choice(vertices.shape[0], int(vertices.shape[0] * reduction_rate), replace=False)
        vertices = vertices[sample_indices]
        vertices = (vertices, sample_indices)
        colors = colors[sample_indices]

    return vertices, triangles, colors

def sample_volume_mesh(fn, sample_num, offset, trans_matrix=None):
    mesh = trimesh.load(fn)
    mesh.apply_transform(trans_matrix)
    mesh.vertices += offset
    samples = trimesh.sample.volume_mesh(mesh, sample_num)
    return samples
    
def load_point_cloud_ply(fn, scale=1, offset=(0, 0,0), sample_num_points=None, angle_x=0, angle_y=0, angle_z=0):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    colors = np.vstack([plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue']]).T / 255.0
    # random sample
    if sample_num_points is not None:
        indices = np.random.choice(points.shape[0], sample_num_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    print('Loaded, point count:', points.shape)
    points = rotate_mesh(points, angle_x, angle_y, angle_z)
    points = points * scale + offset
    return points, colors

def write_point_cloud(fn, pos_and_color):
    num_particles = len(pos_and_color)
    with open(fn, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar placeholder
end_header
"""
        f.write(str.encode(header))
        f.write(pos_and_color.tobytes())
