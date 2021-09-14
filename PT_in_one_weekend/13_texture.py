import os
import time
import numpy as np
import taichi as ti
from PIL import Image
import imageio
from tqdm import trange 

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
ti.init(device_memory_GB=8, use_unified_memory=False, arch=ti.gpu)

INF = 114514114514

Vec2i = ti.types.vector(2, ti.i32)
Vec2f = ti.types.vector(2, ti.f32)
Vec3f = ti.types.vector(3, ti.f32)
Mat3f = ti.types.matrix(3, 3, ti.f32)

Ray = ti.types.struct(ro=Vec3f, rd=Vec3f, l=Vec3f)
Texture = ti.types.struct(albedo=Vec3f, normal=Vec3f, roughness=ti.f32, metallic=ti.f32)
TextureArea = ti.types.struct(low=Vec2i, high=Vec2i)
Material = ti.types.struct(albedo=Vec3f, roughness=ti.f32, metallic=ti.f32, ior=ti.f32, absorptivity=ti.f32, transparency=ti.i32)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, dir=ti.i32, t=ti.f32, material=Material)

Sphere = ti.types.struct(center=Vec3f, radius=ti.f32, transparency=ti.i32, texture_area=TextureArea)
AABB = ti.types.struct(low=Vec3f, high=Vec3f)

BVHNode = ti.types.struct(left=ti.i32, right=ti.i32, aabb=AABB, data=ti.i32)

resolution = (3000, 2000)
texture_size = (4096, 2048)
environment_size = (2048, 1024)
spp = 8192
batch = 8
propagate_limit = 10
epsilon = 1e-4

image = Vec3f.field()
rays = Ray.field()
hits = HitRecord.field()
textures = Texture.field()
environments = Vec3f.field()
ti.root.dense(ti.ij, resolution).place(image)
ti.root.bitmasked(ti.ijk, (*resolution, 1)).bitmasked(ti.k, batch).place(rays)
ti.root.bitmasked(ti.ijk, (*resolution, 1)).bitmasked(ti.k, batch).place(hits)
ti.root.dense(ti.ij, texture_size).place(textures)
ti.root.dense(ti.ij, environment_size).place(environments)


def load_texture(config):
    texture_albedo = textures.albedo.to_numpy()
    texture_roughness = textures.roughness.to_numpy()
    texture_metallic = textures.metallic.to_numpy()
    texture_normal = textures.normal.to_numpy()
    for record in config:
        file_path = record['file_path']
        area = record['area']
        size = (area.high[0] - area.low[0], area.high[1] - area.low[1])
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
    textures.albedo.from_numpy(texture_albedo)
    textures.roughness.from_numpy(texture_roughness)
    textures.metallic.from_numpy(texture_metallic)
    textures.normal.from_numpy(texture_normal)


def load_environment(config):
    envs = environments.to_numpy()
    for record in config:
        file_path = record['file_path']
        area = record['area']
        env = imageio.imread(file_path)
        env = env.transpose(1, 0, 2)[..., :3]
        env = np.flip(env, 1)
        envs[area.low[0]:area.high[0], area.low[1]:area.high[1]] = env
    environments.from_numpy(envs)


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
def nearest(texture, area, u, v):
    u = ti.cast(u, ti.i32)
    v = ti.cast(v, ti.i32)
    u = area.low[0] + ti.mod(u, (area.high[0] - area.low[0]))
    v = area.low[1] + ti.mod(v, (area.high[1] - area.low[1]))
    res = texture[u, v]
    return res


@ti.func
def bilinear(texture, area, u, v):
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
        self.tree_leaves_field = Sphere.field()
        self.tree_leaves_field_cut = ti.field(ti.i32)
        self.tree_nodes = []
        self.tree_leaves = []
        self.max_depth = 0

    @staticmethod
    def split_node(objects):
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
                objects_left, objects_right, aabb_left, aabb_right = BVHTree.split_node(tree[i]['objects'])
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
class World:
    def __init__(self, objects=[]):
        self.objects = objects
        self.environment = TextureArea.field(shape=())
        self.bvh = BVHTree()

    def add(self, object):
        self.objects.append(object)

    def set_environment(self, area):
        self.environment = area

    def build(self):
        self.bvh.build(self.objects)

    @ti.func
    def hit(self, ray):
        return self.bvh.hit(ray)


@ti.func
def aabb_hit(object, ray):
    invdir = 1 / ray.rd
    i = (object.low - ray.ro) * invdir
    o = (object.high - ray.ro) * invdir
    tmax = ti.max(i, o)
    tmin = ti.min(i, o)
    t1 = ti.min(tmax[0], ti.min(tmax[1], tmax[2]))
    t0 = ti.max(tmin[0], ti.max(tmin[1], tmin[2]))
    
    return t1 > t0 and t1 > 0


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
        u = (theta / np.pi + 1) / 2 * (object.texture_area.high[0] - object.texture_area.low[0])
        v = (phi / np.pi + 0.5) * (object.texture_area.high[1] - object.texture_area.low[1])
        texture = bilinear(textures, object.texture_area, 2 * u, 1 * v)
        N_offset = texture.normal - Vec3f([0,0,1])
        record.normal = (N_offset[0] * T + N_offset[1] * B + (1 + N_offset[2]) * N).normalized()
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
    if ray.rd.dot(res.normal) > 0:
        res.normal = -res.normal
        res.material.ior = 1 / res.material.ior
        res.material.absorptivity = 0
    return res


@ti.func
def triangle_list_hit(objects, vertices, start, end, ray):
    res = HitRecord(0.0)
    res.t = -1
    for i in range(start, end):
        record = sphere_hit(objects[i], ray)
        if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
    if ray.rd.dot(res.normal) > 0:
        res.normal = -res.normal
        res.material.ior = 1 / res.material.ior
        res.material.absorptivity = 0
    return res


@ti.func
def environment_color(ray, area):
    phi = ti.asin(ray.rd[1])
    theta = ti.atan2(-ray.rd[0], -ray.rd[2])
    u = (theta / np.pi + 1) / 2 * (area.high[0] - area.low[0])
    v = (phi / np.pi + 0.5) * (area.high[1] - area.low[1])
    color = bilinear(environments, area, u, v)
    return color


@ti.kernel
def propagate_once(rays: ti.template(), hits: ti.template()):
    for i, j, k in rays:
        record = world.hit(rays[i, j, k])
        if record.t >= 0:
            hits[i, j, k] = record
        else:
            image[i, j] += environment_color(rays[i, j, k], world.environment) * rays[i, j, k].l / spp


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
        image[i, j] = image[i, j]**(1/2.2)


def render():
    global rays, hits
    image.fill(0)
    for _ in trange(spp // batch):
        camera.get_rays(rays)
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
    {'file_path': './textures/soft-blanket', 'area': TextureArea(low=Vec2i([0, 0]), high=Vec2i([2048, 2048]))}
])

load_environment([
    {'file_path': './textures/cayley_interior_2k.exr', 'area': TextureArea(low=Vec2i([0, 0]), high=Vec2i([2048, 1024]))}
])

camera = Camera(resolution)
camera.set_fov(30)
# camera.set_len(10, 0.1)
camera.set_position(Vec3f([13, 2, 3]) * 0.3)
camera.look_at(Vec3f([0, 0, 0]))
world = World()
sphere = Sphere(center=Vec3f([0, 0, 0]), radius=1, transparency=0, texture_area=TextureArea(low=Vec2i([0, 0]), high=Vec2i([2048, 2048])))
world.add(sphere)
world.set_environment(TextureArea(low=Vec2i([0, 0]), high=Vec2i([2048, 1024])))
world.build()

render()
ti.imwrite(image, '13_texture.png')

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
#     camera.look_at(Vec3f([0, 0, 0]))
#     render()
#     gui.set_image(image)
#     gui.show()
#    t_last = t
