import os
import pickle
import time
import numpy as np
import taichi as ti
from PIL import Image
import imageio
from tqdm import trange 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ti.init(device_memory_GB=8, use_unified_memory=False, arch=ti.gpu)

INF = 114514114514

Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3i = ti.types.vector(3, ti.i32)
Vec3f = ti.types.vector(3, ti.f32)
Mat3f = ti.types.matrix(3, 3, ti.f32)

Ray = ti.types.struct(ro=Vec3f, rd=Vec3f, l=Vec3f)
Texture = ti.types.struct(albedo=Vec3f, normal=Vec3f, roughness=ti.f32, metallic=ti.f32)
TextureArea = ti.types.struct(low=Vec2i, high=Vec2i)
Material = ti.types.struct(albedo=Vec3f, roughness=ti.f32, metallic=ti.f32, ior=ti.f32, absorptivity=ti.f32, transparency=ti.i32)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, dir=ti.i32, t=ti.f32, material=Material)

Vertex = ti.types.struct(p=Vec3f, n=Vec3f, t=Vec2f)
Triangle = ti.types.struct(a=Vertex, b=Vertex, c=Vertex, texture_id=ti.i32)
FaceVertex = ti.types.struct(p=ti.i32, n=ti.i32, t=ti.i32)
Face = ti.types.struct(a=FaceVertex, b=FaceVertex, c=FaceVertex, texture_id=ti.i32)
Sphere = ti.types.struct(center=Vec3f, radius=ti.f32, transparency=ti.i32, texture_id=ti.i32)
AABB = ti.types.struct(low=Vec3f, high=Vec3f)

BVHNode = ti.types.struct(left=ti.i32, right=ti.i32, aabb=AABB, data=ti.i32)

resolution = (600, 400)
texture_size = (2048 * 5, 2048)
texture_maxnum = 32
environment_size = (2048, 1024)
environment_maxnum = 32
spp = 8192
batch = 128
propagate_limit = 10
epsilon = 1e-8

image = Vec3f.field()
rays = Ray.field()
hits = HitRecord.field()
textures = Texture.field()
textures_info = TextureArea.field()
environments = Vec3f.field()
environments_info = TextureArea.field()
ti.root.dense(ti.ij, resolution).place(image)
ti.root.bitmasked(ti.ijk, (*resolution, 1)).bitmasked(ti.k, batch).place(rays)
ti.root.bitmasked(ti.ijk, (*resolution, 1)).bitmasked(ti.k, batch).place(hits)
ti.root.dense(ti.ij, texture_size).place(textures)
ti.root.dense(ti.i, texture_maxnum).place(textures_info)
ti.root.dense(ti.ij, environment_size).place(environments)
ti.root.dense(ti.i, environment_maxnum).place(environments_info)



def load_texture(config):
    texture_albedo = textures.albedo.to_numpy()
    texture_roughness = textures.roughness.to_numpy()
    texture_metallic = textures.metallic.to_numpy()
    texture_normal = textures.normal.to_numpy()
    for record in config:
        file_path = record['file_path']
        area = record['area']
        id = record['id']
        size = (area.high[0] - area.low[0], area.high[1] - area.low[1])
        if os.path.exists(file_path):
            albedo = Image.open(file_path)
            albedo = albedo.resize(size, Image.ANTIALIAS)
            albedo = np.array(albedo).transpose(1, 0, 2)[..., :3] / 255.0
            albedo = np.flip(albedo, 1)
            roughness = 1
            metallic = 0
            normal = np.array([0.5, 0.5, 1])
        else:
            albedo = Image.open(file_path + '_albedo.png')
            albedo = albedo.resize(size, Image.ANTIALIAS)
            albedo = np.array(albedo).transpose(1, 0, 2)[..., :3] / 255.0
            albedo = np.flip(albedo, 1)
            roughness = Image.open(file_path + '_roughness.png').convert('L')
            roughness = roughness.resize(size, Image.ANTIALIAS)
            roughness = np.array(roughness).transpose(1, 0) / 255.0
            roughness = np.flip(roughness, 1)
            metallic = Image.open(file_path + '_metallic.png').convert('L')
            metallic = metallic.resize(size, Image.ANTIALIAS)
            metallic = np.array(metallic).transpose(1, 0) / 255.0
            metallic = np.flip(metallic, 1)
            normal = Image.open(file_path + '_normal.png')
            normal = normal.resize(size, Image.ANTIALIAS)
            normal = np.array(normal).transpose(1, 0, 2)[..., :3] / 255.0
            normal = np.flip(normal, 1)

        albedo = albedo**2.2
        roughness = roughness**2
        metallic = metallic**2
        normal = normal * 2 - 1

        texture_albedo[area.low[0]:area.high[0], area.low[1]:area.high[1]] = albedo
        texture_roughness[area.low[0]:area.high[0], area.low[1]:area.high[1]] = roughness
        texture_metallic[area.low[0]:area.high[0], area.low[1]:area.high[1]] = metallic
        texture_normal[area.low[0]:area.high[0], area.low[1]:area.high[1]] = normal
        textures_info[id] = area

    textures.albedo.from_numpy(texture_albedo)
    textures.roughness.from_numpy(texture_roughness)
    textures.metallic.from_numpy(texture_metallic)
    textures.normal.from_numpy(texture_normal)


def load_environment(config):
    envs = environments.to_numpy()
    for record in config:
        file_path = record['file_path']
        area = record['area']
        id = record['id']
        env = imageio.imread(file_path)
        env = env.transpose(1, 0, 2)[..., :3]
        env = np.flip(env, 1)
        envs[area.low[0]:area.high[0], area.low[1]:area.high[1]] = env
        environments_info[id] = area
    environments.from_numpy(envs)


def load_obj(file_path, texture_start_id, flip_textcoord=False):
    dir_path = os.path.dirname(file_path)
    positions = []
    normals = []
    texture_coords = []
    indices = []
    textures = dict()
    usemtl = None
    with open(file_path, 'r') as obj:
        lines = obj.readlines()
    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == 'mtllib':
            mtl_name = None
            mtl_filepath = os.path.join(dir_path, line[1])
            with open(mtl_filepath, 'r') as mtl:
                mtl_lines = mtl.readlines()
            for mtl_line in mtl_lines:
                mtl_line = mtl_line.split()
                if len(mtl_line) == 0:
                    continue
                if mtl_line[0] == 'newmtl':
                    mtl_name = mtl_line[1]
                elif mtl_line[0] == 'map_Kd':
                    mtl_filepath = os.path.join(dir_path, mtl_line[1])
                    textures[mtl_name] = {'file_path': mtl_filepath, 'id': texture_start_id}
                    texture_start_id += 1
        elif line[0] == 'v':
            positions.append(Vec3f([float(line[1]), float(line[2]), float(line[3])]))
        elif line[0] == 'vn':
            normals.append(Vec3f([float(line[1]), float(line[2]), float(line[3])]))
        elif line[0] == 'vt':
            u = float(line[1])
            v = float(line[2])
            if flip_textcoord:
                v = 1- v
            texture_coords.append(Vec2f([u, v]))
        elif line[0] == 'usemtl':
            usemtl = line[1]
        elif line[0] == 'f':
            for i in range(1, 4):
                line[i] = line[i].split('/')
            indices.append(Face(
                a=FaceVertex(p=int(line[1][0]) - 1, n=int(line[1][2]) - 1, t=int(line[1][1]) - 1),
                b=FaceVertex(p=int(line[2][0]) - 1, n=int(line[2][2]) - 1, t=int(line[2][1]) - 1),
                c=FaceVertex(p=int(line[3][0]) - 1, n=int(line[3][2]) - 1, t=int(line[3][1]) - 1),
                texture_id = textures[usemtl]['id']
            ))
    return positions, normals, texture_coords, indices, list(textures.values())


@ti.func
def array_get(a: ti.template(), i):
    ret = a[0]
    for j in ti.static(range(len(a))):
        if i == j:
            ret = a[j]
    return ret


@ti.func
def array_set(a: ti.template(), i, val):
    for j in ti.static(range(len(a))):
        if i == j:
            a[j] = val


@ti.func
def nearest(texture, info, id, u, v):
    area = info[id]
    u = u * (area.high[0] - area.low[0])
    v = v * (area.high[1] - area.low[1])
    u = ti.cast(u, ti.i32)
    v = ti.cast(v, ti.i32)
    u = area.low[0] + ti.mod(u, (area.high[0] - area.low[0]))
    v = area.low[1] + ti.mod(v, (area.high[1] - area.low[1]))
    res = texture[u, v]
    return res


@ti.func
def bilinear(texture, info, id, u, v):
    area = info[id]
    u = u * (area.high[0] - area.low[0])
    v = v * (area.high[1] - area.low[1])
    u = u - 0.5
    v = v - 0.5
    l = ti.cast(u, ti.i32)
    r = l + 1
    b = ti.cast(v, ti.i32)
    t = b + 1
    lb = (r - u) * (t - v)
    lt = (r - u) * (v - b)
    rb = (u - l) * (t - v)
    rt = (u - l) * (v - b)
    l = area.low[0] + ti.mod(l, (area.high[0] - area.low[0]))
    r = area.low[0] + ti.mod(r, (area.high[0] - area.low[0]))
    b = area.low[1] + ti.mod(b, (area.high[0] - area.low[0]))
    t = area.low[1] + ti.mod(t, (area.high[0] - area.low[0]))
    res = lb * texture[l, b] + lt * texture[l, t] + rb * texture[r, b] + rt * texture[r, t]
    return res


@ti.pyfunc
def rotate(yaw, pitch, roll=0):
    yaw_trans = Mat3f([
        [ ti.cos(yaw), 0, ti.sin(yaw)],
        [           0, 1,           0],
        [-ti.sin(yaw), 0, ti.cos(yaw)],
    ])
    pitch_trans = Mat3f([
        [1,             0,              0],
        [0, ti.cos(pitch), -ti.sin(pitch)],
        [0, ti.sin(pitch),  ti.cos(pitch)],
    ])
    roll_trans = Mat3f([
        [ti.cos(roll), -ti.sin(roll), 0],
        [ti.sin(roll),  ti.cos(roll), 0],
        [           0,             0, 1],
    ])
    return yaw_trans @ pitch_trans @ roll_trans


@ti.func
def cal_reflectivity_metal(dir, normal, material):
    F0 = material.albedo
    F = F0 + (1 - F0) * (1 + (normal.dot(dir)))**5
    return F


@ti.func
def cal_reflectivity_dielectirc(dir, normal, material):
    F0 = ((material.ior - 1) / (material.ior + 1))**2
    F = F0 + (1 - F0) * (1 + (normal.dot(dir)))**5
    return F


@ti.func
def sample_at_sphere():
    z = 1 - 2 * ti.random(ti.f32)
    r = ti.sqrt(1 - z**2)
    theta = 2 * np.pi * ti.random(ti.f32)
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    return Vec3f([x, y, z])

@ti.func
def sample_in_sphere():
    r = ti.random(ti.f32)**(1.0 / 3.0)
    theta = 2 * np.pi * ti.random(ti.f32)
    phi = ti.acos(ti.random(ti.f32) * 2 - 1)
    x = r * ti.cos(theta) * ti.sin(phi)
    y = r * ti.sin(theta) * ti.sin(phi)
    z = r * ti.cos(phi)
    return Vec3f([x, y, z])

@ti.func
def sample_in_disk():
    r = ti.sqrt(ti.random(ti.f32))
    theta = 2 * np.pi * ti.random(ti.f32)
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    return Vec2f([x, y])


@ti.func
def sample_diffuse(normal):
    s = sample_at_sphere()
    return (normal + s).normalized()


@ti.func
def sample_reflect(dir, normal, material):
    s = sample_in_sphere()
    k = -dir.dot(normal)
    new_dir = dir + 2 * k * normal
    return (new_dir + material.roughness * s).normalized()


@ti.func
def sample_refract(dir, normal, material):
    s = sample_in_sphere()
    k = dir.dot(normal)
    r_out_perp = (dir - k * normal) / material.ior
    r_out_perp_len2 = r_out_perp.dot(r_out_perp)
    if r_out_perp_len2 > 1 : r_out_perp_len2 = 1
    k = ti.sqrt(1.0 - r_out_perp_len2)
    r_out_parallel = -k * normal
    new_dir = r_out_perp + r_out_parallel
    return (new_dir + material.roughness * s).normalized()


@ti.data_oriented
class Camera:
    def __init__(self, resolution, fov=60, focal_length=1, aperture=0):
        self.resolution = Vec2i.field(shape=())
        self.fov = ti.field(ti.f32, shape=())
        self.focal_length = ti.field(ti.f32, shape=())
        self.aperture = ti.field(ti.f32, shape=())
        self.position = Vec3f.field(shape=())
        self.front_axis = Vec3f.field(shape=())
        self.right_axis = Vec3f.field(shape=())
        self.up_axis = Vec3f.field(shape=())

        self.resolution[None] = resolution
        self.fov[None] = float(fov)
        self.focal_length[None] = float(focal_length)
        self.aperture[None] = float(aperture)
        self.position[None] = Vec3f(0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self

    def set_position(self, position):
        self.position[None] = position

    def set_fov(self, fov):
        self.fov[None] = float(fov)

    def set_len(self, focal_length=1, aperture=0):
        self.focal_length[None] = float(focal_length)
        self.aperture[None] = float(aperture)

    def set_direction(self, yaw, pitch, roll=0):
        self.yaw = float(yaw) * np.pi / 180
        self.pitch = float(pitch) * np.pi / 180
        self.roll = float(roll) * np.pi / 180
        self.update_coord()

    def look_at(self, target, roll=0):
        position = Vec3f([self.position[None][0], self.position[None][1], self.position[None][2]])
        dir = (target - position).normalized()
        self.yaw = ti.atan2(-dir[0], -dir[2])
        self.pitch = ti.asin(dir[1])
        self.roll = float(roll) * np.pi / 180
        self.update_coord()

    def update_coord(self):
        trans = rotate(self.yaw, self.pitch, self.roll)
        self.front_axis[None] = trans @ Vec3f([0.0, 0.0, -1.0])
        self.right_axis[None] = trans @ Vec3f([1.0, 0.0, 0.0])
        self.up_axis[None] = trans @ Vec3f([0.0, 1.0, 0.0])

    def move_front(self, d):
        self.position[None][0] += d * self.front_axis[None][0]
        self.position[None][1] += d * self.front_axis[None][1]
        self.position[None][2] += d * self.front_axis[None][2]

    def move_right(self, d):
        self.position[None][0] += d * self.right_axis[None][0]
        self.position[None][1] += d * self.right_axis[None][1]
        self.position[None][2] += d * self.right_axis[None][2]

    def move_up(self, d):
        self.position[None][1] += d

    def rotate(self, yaw, pitch, roll=0):
        self.yaw += yaw
        self.pitch += pitch
        self.pitch = max(-np.pi + epsilon, min(np.pi - epsilon, self.pitch))
        self.roll += roll
        self.update_coord()

    @ti.kernel
    def get_rays_fast(self, rays: ti.template()):
        width = self.resolution[None][0]
        height = self.resolution[None][1]

        ratio = height / width
        view_width = 2 * ti.tan(self.fov[None] * np.pi / 180)
        view_height = view_width * ratio
        
        for i, j, k in ti.ndrange(width, height, rays.shape[2]):
            target = self.front_axis[None] + (i / width - 0.5) * view_width * self.right_axis[None] + (j / height - 0.5) * view_height * self.up_axis[None]
            rays[i, j, k].ro = self.position[None]
            rays[i, j, k].rd = target.normalized()
            rays[i, j, k].l = Vec3f([1.0, 1.0, 1.0])    

    @ti.kernel
    def get_rays(self, rays: ti.template()):
        width = self.resolution[None][0]
        height = self.resolution[None][1]

        ratio = height / width
        view_width = 2 * ti.tan(self.fov[None] * np.pi / 180)
        view_height = view_width * ratio
        
        for i, j, k in ti.ndrange(width, height, rays.shape[2]):
            target = self.focal_length[None] * (self.front_axis[None] + ((i + ti.random(ti.f32)) / width - 0.5) * view_width * self.right_axis[None] + ((j + ti.random(ti.f32)) / height - 0.5) * view_height * self.up_axis[None])
            sample = sample_in_disk()
            origin = self.aperture[None] / 2.0 * (sample[0] * self.right_axis[None] + sample[1] * self.up_axis[None])
            rays[i, j, k].ro = self.position[None] + origin
            rays[i, j, k].rd = (target - origin).normalized()
            rays[i, j, k].l = Vec3f([1.0, 1.0, 1.0])


@ti.data_oriented
class BVHTree:
    def __init__(self):
        self.tree_nodes_field = BVHNode.field()
        self.tree_leaves_field = None
        self.tree_leaves_field_cut = ti.field(ti.i32)
        self.tree_nodes = []
        self.tree_leaves = []
        self.max_depth = 0
        self.need_dump = ['tree_nodes_field', 'tree_leaves_field', 'tree_leaves_field_cut']

    def print(self, i=0, depth=0):
        nodes = self.tree_nodes
        leaves = self.tree_leaves
        if i >= 0:
            if nodes[i].data >= 0:
                print('  ' * depth, 'AABB: ', nodes[i].aabb, '  OBJS: ', len(leaves[nodes[i].data]))
            else:
                print('  ' * depth, 'AABB: ', nodes[i].aabb)
            self.print(nodes[i].left, depth + 1)
            self.print(nodes[i].right, depth + 1)

    def build_field(self):
        leaves_num = 0
        for i in range(len(self.tree_leaves)):
            leaves_num += len(self.tree_leaves[i])

        ti.root.dense(ti.i, len(self.tree_nodes)).place(self.tree_nodes_field)
        ti.root.dense(ti.i, len(self.tree_leaves) + 1).place(self.tree_leaves_field_cut)
        ti.root.dense(ti.i, leaves_num).place(self.tree_leaves_field)

        for i in range(len(self.tree_nodes)):
            self.tree_nodes_field[i] = self.tree_nodes[i]

        for i in range(len(self.tree_leaves)):
            self.tree_leaves_field_cut[i + 1] = self.tree_leaves_field_cut[i] + len(self.tree_leaves[i])

        for i in range(len(self.tree_leaves)):
            for j in range(len(self.tree_leaves[i])):
                self.tree_leaves_field[self.tree_leaves_field_cut[i] + j] = self.tree_leaves[i][j]

    def dump(self):
        data = {'max_depth': self.max_depth}
        for name in self.need_dump:
            data[name] = {'data': getattr(self, name).to_numpy(), 'shape': list(getattr(self, name).shape)}
        return data

    def load(self, data):
        self.max_depth = data['max_depth']
        data.pop('max_depth')
        layout = [None, ti.i, ti.ik, ti.ijk]
        for name, val in data.items():
            ti.root.dense(layout[len(val['shape'])], val['shape']).place(getattr(self, name))
            getattr(self, name).from_numpy(val['data'])


@ti.data_oriented
class SphereBVHTree(BVHTree):
    def __init__(self):
        super().__init__()
        self.tree_leaves_field = Sphere.field()

    def split_node(self, objects):
        aabbs = [[[AABB(0.0), AABB(0.0)] for _ in range(len(objects) - 1)] for _ in range(3)]
        sorted_objects = [None, None, None]
        min_cost = INF
        min_axis = None
        min_idx = None

        for axis in range(3):
            sorted_objects[axis] = objects.copy()
            sorted_objects[axis].sort(key=lambda x: x.center[axis])

            low=Vec3f([INF, INF, INF])
            high=Vec3f([-INF, -INF, -INF])
            for i in range(len(sorted_objects[axis]) - 1):
                low = ti.min(low, sorted_objects[axis][i].center - sorted_objects[axis][i].radius)
                high = ti.max(high, sorted_objects[axis][i].center + sorted_objects[axis][i].radius)
                aabbs[axis][i][0].low = low
                aabbs[axis][i][0].high = high

            low=Vec3f([INF, INF, INF])
            high=Vec3f([-INF, -INF, -INF])
            for i in range(len(sorted_objects[axis]) - 1, 0, -1):
                low = ti.min(low, sorted_objects[axis][i].center - sorted_objects[axis][i].radius)
                high = ti.max(high, sorted_objects[axis][i].center + sorted_objects[axis][i].radius)
                aabbs[axis][i - 1][1].low = low
                aabbs[axis][i - 1][1].high = high

            for i in range(len(sorted_objects[axis]) - 1):
                size0 = aabbs[axis][i][0].high - aabbs[axis][i][0].low
                size1 = aabbs[axis][i][1].high - aabbs[axis][i][1].low
                area0 = size0[0] * size0[1] + size0[1] * size0[2] + size0[2] * size0[0]
                area1 = size1[0] * size1[1] + size1[1] * size1[2] + size1[2] * size1[0]
                num0 = i + 1
                num1 = len(sorted_objects[axis]) - num0
                cost = num0 * area0 + num1 * area1
                if cost < min_cost:
                    min_cost = cost
                    min_axis = axis
                    min_idx = i

        return sorted_objects[min_axis][:min_idx + 1], sorted_objects[min_axis][min_idx + 1:], aabbs[min_axis][min_idx][0], aabbs[min_axis][min_idx][1] 

    def build(self, objects, max_depth=8, max_leave_objects=4):
        self.max_depth = max_depth
        low = Vec3f([INF, INF, INF])
        high = Vec3f([-INF, -INF, -INF])
        tree = []
        for object in objects:
            low = ti.min(low, object.center - object.radius)
            high = ti.max(high, object.center + object.radius)
        self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=AABB(low=low, high=high), data=-1))
        tree.append({'depth': 0, 'objects': objects})
        i = 0
        while i < len(tree):
            if tree[i]['depth'] < max_depth and len(tree[i]['objects']) > max_leave_objects:
                objects_left, objects_right, aabb_left, aabb_right = self.split_node(tree[i]['objects'])
                self.tree_nodes[i].left = len(tree)
                self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=aabb_left, data=-1))
                tree.append({'depth': tree[i]['depth'] + 1, 'objects': objects_left})
                self.tree_nodes[i].right = len(tree)
                self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=aabb_right, data=-1))
                tree.append({'depth': tree[i]['depth'] + 1, 'objects': objects_right})
            else:
                self.tree_nodes[i].data = len(self.tree_leaves)
                self.tree_leaves.append(tree[i]['objects'])
            i += 1

        self.print()
        self.build_field()

    @ti.func
    def hit(self, ray):
        res = HitRecord(0.0)
        res.t = -1
        stack = ti.Vector.zero(ti.i32, self.max_depth + 1)
        stack_p = 0
        array_set(stack, stack_p, 0)
        while stack_p >= 0:
            cur_node = array_get(stack, stack_p)
            if aabb_hit(self.tree_nodes_field[cur_node].aabb, ray):
                if self.tree_nodes_field[cur_node].data >= 0:   
                    record = sphere_list_hit(self.tree_leaves_field, self.tree_leaves_field_cut[self.tree_nodes_field[cur_node].data], self.tree_leaves_field_cut[self.tree_nodes_field[cur_node].data + 1], ray)
                    if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
                    stack_p -= 1
                else:
                    array_set(stack, stack_p, self.tree_nodes_field[cur_node].left)
                    stack_p += 1
                    array_set(stack, stack_p, self.tree_nodes_field[cur_node].right)
            else:
                stack_p -= 1
        return res


@ti.data_oriented
class MeshBVHTree(BVHTree):
    def __init__(self):
        super().__init__()
        self.positions_field = Vec3f.field()
        self.positions = []
        self.normals_field = Vec3f.field()
        self.normals = []
        self.texture_coords_field = Vec2f.field()
        self.texture_coords = []
        self.tree_leaves_field = Face.field()
        self.need_dump.append('positions_field')
        self.need_dump.append('normals_field')
        self.need_dump.append('texture_coords_field')

    def split_node(self, objects):
        aabbs = [[[AABB(0.0), AABB(0.0)] for _ in range(len(objects) - 1)] for _ in range(3)]
        sorted_objects = [None, None, None]
        min_cost = INF
        min_axis = None
        min_idx = None

        for axis in range(3):
            sorted_objects[axis] = objects.copy()
            sorted_objects[axis].sort(key=lambda x: (self.positions[x.a.p] + self.positions[x.b.p] + self.positions[x.c.p]) / 3)

            low=Vec3f([INF, INF, INF])
            high=Vec3f([-INF, -INF, -INF])
            for i in range(len(sorted_objects[axis]) - 1):
                low = ti.min(low, ti.min(self.positions[sorted_objects[axis][i].a.p], ti.min(self.positions[sorted_objects[axis][i].b.p] ,self.positions[sorted_objects[axis][i].c.p])))
                high = ti.max(high, ti.max(self.positions[sorted_objects[axis][i].a.p], ti.max(self.positions[sorted_objects[axis][i].b.p] ,self.positions[sorted_objects[axis][i].c.p])))
                aabbs[axis][i][0].low = low
                aabbs[axis][i][0].high = high

            low=Vec3f([INF, INF, INF])
            high=Vec3f([-INF, -INF, -INF])
            for i in range(len(sorted_objects[axis]) - 1, 0, -1):
                low = ti.min(low, ti.min(self.positions[sorted_objects[axis][i].a.p], ti.min(self.positions[sorted_objects[axis][i].b.p] ,self.positions[sorted_objects[axis][i].c.p])))
                high = ti.max(high, ti.max(self.positions[sorted_objects[axis][i].a.p], ti.max(self.positions[sorted_objects[axis][i].b.p] ,self.positions[sorted_objects[axis][i].c.p])))
                aabbs[axis][i - 1][1].low = low
                aabbs[axis][i - 1][1].high = high

            for i in range(len(sorted_objects[axis]) - 1):
                size0 = aabbs[axis][i][0].high - aabbs[axis][i][0].low
                size1 = aabbs[axis][i][1].high - aabbs[axis][i][1].low
                area0 = size0[0] * size0[1] + size0[1] * size0[2] + size0[2] * size0[0]
                area1 = size1[0] * size1[1] + size1[1] * size1[2] + size1[2] * size1[0]
                num0 = i + 1
                num1 = len(sorted_objects[axis]) - num0
                cost = num0 * area0 + num1 * area1
                if cost < min_cost:
                    min_cost = cost
                    min_axis = axis
                    min_idx = i

        return sorted_objects[min_axis][:min_idx + 1], sorted_objects[min_axis][min_idx + 1:], aabbs[min_axis][min_idx][0], aabbs[min_axis][min_idx][1] 

    def build(self, positions, normals, texture_coords, indices, max_depth=16, max_leave_objects=4):
        self.positions = positions
        self.normals = normals
        self.texture_coords = texture_coords
        self.max_depth = max_depth
        low = Vec3f([INF, INF, INF])
        high = Vec3f([-INF, -INF, -INF])
        tree = []
        for triangle in indices:
            low = ti.min(low, ti.min(positions[triangle.a.p], ti.min(positions[triangle.b.p] ,positions[triangle.c.p])))
            high = ti.max(high, ti.max(positions[triangle.a.p], ti.max(positions[triangle.b.p] ,positions[triangle.c.p])))
        self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=AABB(low=low, high=high), data=-1))
        tree.append({'depth': 0, 'objects': indices})
        i = 0
        while i < len(tree):
            if tree[i]['depth'] < max_depth and len(tree[i]['objects']) > max_leave_objects:
                objects_left, objects_right, aabb_left, aabb_right = self.split_node(tree[i]['objects'])
                self.tree_nodes[i].left = len(tree)
                self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=aabb_left, data=-1))
                tree.append({'depth': tree[i]['depth'] + 1, 'objects': objects_left})
                self.tree_nodes[i].right = len(tree)
                self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=aabb_right, data=-1))
                tree.append({'depth': tree[i]['depth'] + 1, 'objects': objects_right})
            else:
                self.tree_nodes[i].data = len(self.tree_leaves)
                self.tree_leaves.append(tree[i]['objects'])
            i += 1

        self.print()
        ti.root.dense(ti.i, len(self.positions)).place(self.positions_field)
        ti.root.dense(ti.i, len(self.normals)).place(self.normals_field)
        ti.root.dense(ti.i, len(self.texture_coords)).place(self.texture_coords_field)
        self.build_field()
        for i in range(len(self.positions)):
            self.positions_field[i] = self.positions[i]
        for i in range(len(self.normals)):
            self.normals_field[i] = self.normals[i]
        for i in range(len(self.texture_coords)):
            self.texture_coords_field[i] = self.texture_coords[i]

    @ti.func
    def hit(self, ray):
        res = HitRecord(0.0)
        res.t = -1
        stack = ti.Vector.zero(ti.i32, self.max_depth + 1)
        stack_p = 0
        array_set(stack, stack_p, 0)
        while stack_p >= 0:
            cur_node = array_get(stack, stack_p)
            if aabb_hit(self.tree_nodes_field[cur_node].aabb, ray):
                if self.tree_nodes_field[cur_node].data >= 0:   
                    record = triangle_list_hit(
                        self.tree_leaves_field, self.positions_field, self.normals_field, self.texture_coords_field,
                        self.tree_leaves_field_cut[self.tree_nodes_field[cur_node].data], self.tree_leaves_field_cut[self.tree_nodes_field[cur_node].data + 1], ray
                    )
                    if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
                    stack_p -= 1
                else:
                    array_set(stack, stack_p, self.tree_nodes_field[cur_node].left)
                    stack_p += 1
                    array_set(stack, stack_p, self.tree_nodes_field[cur_node].right)
            else:
                stack_p -= 1
        return res


@ti.data_oriented
class World:
    def __init__(self):
        self.spheres = []
        self.meshes = []
        self.environment = None
        self.spheres_bvh = None
        self.meshes_bvhs = []

    def add_mesh(self, positions, normals, texture_coords, indices):
        self.meshes.append({'positions': positions, 'normals': normals, 'texture_coords': texture_coords, 'indices': indices})

    def add_sphere(self, sphere):
        self.spheres.append(sphere)

    def set_environment(self, id):
        self.environment = id

    def build(self):
        for mesh in self.meshes:
            meshes_bvh = MeshBVHTree()
            meshes_bvh.build(mesh['positions'], mesh['normals'], mesh['texture_coords'], mesh['indices'])
            self.meshes_bvhs.append(meshes_bvh)
        if len(self.spheres) > 0:
            self.spheres_bvh = SphereBVHTree()
            self.spheres_bvh.build(self.spheres)

    def save(self, filename):
        data = {'meshes_bvhs': [], 'environment': self.environment}
        if self.spheres_bvh is not None:
            data['spheres_bvh'] = self.spheres_bvh.dump()
        for meshes_bvh in self.meshes_bvhs:
            data['meshes_bvhs'].append(meshes_bvh.dump())
        np.save(filename, data)

    def load(self, filename):
        data = np.load(filename, allow_pickle=True).item()
        self.environment = data['environment']
        if 'spheres_bvh' in data:
            self.spheres_bvh = SphereBVHTree()
            self.spheres_bvh.load(data['spheres_bvh'])
        for meshes_bvh_data in data['meshes_bvhs']:
            meshes_bvh = MeshBVHTree()
            meshes_bvh.load(meshes_bvh_data)
            self.meshes_bvhs.append(meshes_bvh)

    @ti.func
    def hit(self, ray):
        res = HitRecord(0.0)
        res.t = -1
        if ti.static(self.spheres_bvh is not None):
            res = self.spheres_bvh.hit(ray)
        i = 0
        for i in ti.static(range(len(self.meshes_bvhs))):
            record = self.meshes_bvhs[i].hit(ray)
            if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
        return res


@ti.func
def aabb_hit(object, ray):
    invdir = 1 / ray.rd
    i = (object.low - ray.ro) * invdir
    o = (object.high - ray.ro) * invdir
    tmax = ti.max(i, o)
    tmin = ti.min(i, o)
    t1 = ti.min(tmax[0], ti.min(tmax[1], tmax[2]))
    t0 = ti.max(tmin[0], ti.max(tmin[1], tmin[2]))
    
    return t1 > t0 - epsilon and t1 > 0


@ti.func
def sphere_hit(object, ray):
    oc = ray.ro - object.center
    a = 1
    b = 2.0 * oc.dot(ray.rd)
    c = oc.dot(oc) - object.radius**2
    discriminant = b**2 - 4 * a * c
    record = HitRecord(0.0)
    record.t = -1
    if discriminant >= 0:
        sqrt_discriminant = ti.sqrt(discriminant)
        record.t = (-b - sqrt_discriminant) / (2.0 * a)
        if record.t < epsilon and object.transparency:
            record.t = (-b + sqrt_discriminant) / (2.0 * a)
        record.point = ray.ro + record.t * ray.rd
        N = (record.point - object.center).normalized()
        r = ti.sqrt(N[0]**2 + N[2]**2)
        T = Vec3f(N[2] / r, 0, -N[0] / r)
        B = Vec3f(N[0] * N[1], -r, N[2] * N[1])
        phi = ti.asin(N[1])
        theta = ti.atan2(-N[0], -N[2])
        u = (theta / np.pi + 1) / 2
        v = (phi / np.pi + 0.5)
        texture = bilinear(textures, textures_info, object.texture_id, 2 * u, 1 * v)
        N_coord = texture.normal
        record.normal = (N_coord[0] * T + N_coord[1] * B + N_coord[2] * N).normalized()
        record.material.albedo = texture.albedo
        record.material.roughness = texture.roughness
        record.material.metallic = texture.metallic
        record.material.ior = 1.5
        record.material.absorptivity = 0.5
        record.material.transparency = object.transparency
    return record


@ti.func
def sphere_list_hit(objects, start, end, ray):
    res = HitRecord(0.0)
    res.t = -1
    for i in range(start, end):
        record = sphere_hit(objects[i], ray)
        if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
    return res


@ti.func
def triangle_hit(object, ray):
    record = HitRecord(0.0)
    record.t = -1
    p1 = object.a.p
    p2 = object.b.p
    p3 = object.c.p
    o = ray.ro
    d = ray.rd
    N = (p2 - p1).cross(p3 - p1).normalized()
    # 距离
    t = (N.dot(p1) - o.dot(N)) / d.dot(N)
    if (t > epsilon):
        # 交点计算
        P = o + d * t
        # 判断交点是否在三角形中
        w1 = (p3 - p2).cross(P - p2).dot(N) / (p3 - p2).cross(p1 - p2).dot(N)
        w2 = (p1 - p3).cross(P - p3).dot(N) / (p1 - p3).cross(p2 - p3).dot(N)
        w3 = 1 - w1 - w2
        hit = (w1 > 0 and w2 > 0 and w3 > 0)
        # 命中，封装返回结果
        if hit:
            record.t = t
            record.point = P
            # N = (w1 * object.a.n + w2 * object.b.n + w3 * object.c.n).normalized()
            du1 = object.b.t[0] - object.a.t[0]
            du2 = object.c.t[0] - object.a.t[0]
            dv1 = object.b.t[1] - object.a.t[1]
            dv2 = object.c.t[1] - object.a.t[1]
            T = (dv1 * (p3 - p1) - dv2 * (p2 - p1)) / (dv1 * du2 - dv2 * du1 + epsilon)
            T = (T - T.dot(N) * N).normalized()
            B = T.cross(N)
            u = w1 * object.a.t[0] + w2 * object.b.t[0] + w3 * object.c.t[0]
            v = w1 * object.a.t[1] + w2 * object.b.t[1] + w3 * object.c.t[1]
            texture = bilinear(textures, textures_info, object.texture_id, u, v)
            N_coord = texture.normal
            record.normal = (N_coord[0] * T + N_coord[1] * B + N_coord[2] * N).normalized()
            record.material.albedo = texture.albedo
            record.material.roughness = texture.roughness
            record.material.metallic = texture.metallic
            record.material.ior = 1.5
            record.material.absorptivity = 0.5
            record.material.transparency = 0

    return record


@ti.func
def triangle_list_hit(indices, positions, normals, texture_coords, start, end, ray):
    res = HitRecord(0.0)
    res.t = -1
    for i in range(start, end):
        record = triangle_hit(Triangle(
            a=Vertex(p=positions[indices[i].a.p], n=normals[indices[i].a.n], t=texture_coords[indices[i].a.t]),
            b=Vertex(p=positions[indices[i].b.p], n=normals[indices[i].b.n], t=texture_coords[indices[i].b.t]),
            c=Vertex(p=positions[indices[i].c.p], n=normals[indices[i].c.n], t=texture_coords[indices[i].c.t]),
            texture_id=indices[i].texture_id), ray)
        if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
    return res


@ti.func
def environment_color(ray, id):
    phi = ti.asin(ray.rd[1])
    theta = ti.atan2(-ray.rd[0], -ray.rd[2])
    u = (theta / np.pi + 1) / 2
    v = (phi / np.pi + 0.5)
    color = bilinear(environments, environments_info, id, u, v)
    return color


@ti.kernel
def propagate_once(rays: ti.template(), hits: ti.template()):
    for i, j, k in rays:
        record = world.hit(rays[i, j, k])
        if record.t >= 0:
            if rays[i, j, k].rd.dot(record.normal) > 0:
                record.normal = -record.normal
                record.material.ior = 1 / record.material.ior
                record.material.absorptivity = 0
            hits[i, j, k] = record
        else:
            image[i, j] += environment_color(rays[i, j, k], world.environment) * rays[i, j, k].l


@ti.kernel
def gen_secondary_rays(rays: ti.template(), hits: ti.template()):
    for i, j, k in hits:
        if ti.random(ti.f32) < hits[i, j, k].material.metallic:
            F0 = cal_reflectivity_metal(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
            rays[i, j, k].rd = sample_reflect(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
            rays[i, j, k].l = rays[i, j, k].l * F0
        else:
            F0 = cal_reflectivity_dielectirc(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
            if ti.random(ti.f32) > F0:
                if hits[i, j, k].material.transparency:
                    rays[i, j, k].rd = sample_refract(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
                    rays[i, j, k].l = rays[i, j, k].l * hits[i, j, k].material.albedo * (1 - hits[i, j, k].material.absorptivity)
                else:
                    rays[i, j, k].rd = sample_diffuse(hits[i, j, k].normal)
                    rays[i, j, k].l = rays[i, j, k].l * hits[i, j, k].material.albedo * (1 - hits[i, j, k].material.absorptivity)
            else:
                rays[i, j, k].rd = sample_reflect(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
                rays[i, j, k].l = rays[i, j, k].l
        rays[i, j, k].ro = hits[i, j, k].point + hits[i, j, k].normal * 2 * epsilon


@ti.kernel
def gamma_correction():
    for i, j in image:
        image[i, j] = (image[i, j] / spp)**(1/2.2)


def render():
    global rays, hits
    image.fill(0)
    for _ in trange(spp // batch):
        camera.get_rays_fast(rays)
        for i in range(propagate_limit):
            hits.snode.parent(2).deactivate_all()
            propagate_once(rays, hits)
            rays.snode.parent(2).deactivate_all()
            gen_secondary_rays(rays, hits)
    gamma_correction()


@ti.kernel
def test_aabb(rays: ti.template()):
    aabb = AABB(low=Vec3f([0,0,0]), high=Vec3f([1,1,1]))
    for i, j, k in rays:
        record = aabb_hit(aabb, rays[i, j, k])
        if record:
            image[i, j] += Vec3f([1,1,1]) / spp


load_texture([
    {'file_path': './textures/granite-gray-white', 'area': TextureArea(low=Vec2i([0, 0]), high=Vec2i([2048, 2048])), 'id': 0}
])

load_environment([
    {'file_path': './textures/cayley_interior_2k.exr', 'area': TextureArea(low=Vec2i([0, 0]), high=Vec2i([2048, 1024])), 'id': 0}
])

positions, normals, texture_coords, indices, materials = load_obj('./models/Yoimiya_model/Yoimiya.obj', 0, flip_textcoord=True)
texture_configs = []
for i, material in enumerate(materials):
    texture_configs.append({'file_path': material['file_path'], 'area': TextureArea(low=Vec2i([i * 2048, 0]), high=Vec2i([(i + 1) * 2048, 2048])), 'id': material['id']})
load_texture(texture_configs)

camera = Camera(resolution)
camera.set_fov(30)
# camera.set_len(10, 0.1)
camera.set_position(Vec3f([0, 10, -30]))
camera.look_at(Vec3f([0, 10, 0]))
world = World()
# sphere = Sphere(center=Vec3f([0, 0, 0]), radius=1, transparency=0, texture_id=0)
# positions = [Vec3f([0, -1, 1]), Vec3f([0, -1, -1]), Vec3f([0, 1, -1]), Vec3f([0, 1, 1])]
# normals = [Vec3f([1, 0, 0])]
# texture_coords = [Vec2i([0, 0]), Vec2i([1, 0]), Vec2i([1, 1]), Vec2i([0, 1])]
# indices = [
#     Face(a=FaceVertex(p=0, n=0, t=0), b=FaceVertex(p=1, n=0, t=1), c=FaceVertex(p=2, n=0, t=2), texture_id=0),
#     Face(a=FaceVertex(p=0, n=0, t=0), b=FaceVertex(p=2, n=0, t=2), c=FaceVertex(p=3, n=0, t=3), texture_id=0),
# ]
# world.add_sphere(sphere)
# world.add_mesh(positions, normals, texture_coords, indices)
# world.set_environment(0)
# world.build()
# world.save('Yoimiya.world.npy')
world.load('Yoimiya.world.npy')

render()
ti.imwrite(image, '14_mesh.png')

# t_base = time.time()
# t_last = t_base
# velocity = 5

# gui = ti.GUI('SDF Path Tracer', resolution)
# while gui.running:
#     if gui.get_event(ti.GUI.ESCAPE):
#         gui.running = False
#     t = time.time() - t_base
#     dt = t - t_last
#     if gui.is_pressed('w'):
#         camera.move_front(velocity * dt)
#     elif gui.is_pressed('s'):
#         camera.move_front(-velocity * dt)
#     if gui.is_pressed('a'):
#         camera.move_right(-velocity * dt)
#     elif gui.is_pressed('d'):
#         camera.move_right(velocity * dt)
#     if gui.is_pressed(ti.GUI.SPACE):
#         camera.move_up(velocity * dt)
#     elif gui.is_pressed(ti.GUI.SHIFT):
#         camera.move_up(-velocity * dt)
#     mouse = gui.get_cursor_pos()
#     if gui.is_pressed(ti.GUI.LMB):
#         camera.rotate(mouse[0] - mouse_last[0], -mouse[1] + mouse_last[1])
#     render()
#     gui.set_image(image)
#     gui.show()
#     t_last = t
#     mouse_last = mouse
