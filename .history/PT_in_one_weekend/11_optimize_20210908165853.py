import os
import numpy as np
import taichi as ti
from tqdm import trange 

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
ti.init(device_memory_GB=8, use_unified_memory=False, arch=ti.gpu)

INF = 114514

Vec2f = ti.types.vector(2, float)
Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f, l=Vec3f)
Material = ti.types.struct(color=Vec3f, roughness=ti.f32, metalness=ti.i32, ior=ti.f32, absorptivity=ti.f32, transparency=ti.i32)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, dir=ti.i32, t=ti.f32, material=Material)
Sphere = ti.types.struct(center=Vec3f, radius=ti.f32, material=Material)
AABB = ti.types.struct(low=Vec3f, high=Vec3f)
BVHNode = ti.types.struct(left=ti.i32, right=ti.i32, aabb=AABB, data=ti.i32)

resolution = (640, 360)
spp = 128
batch = 128
propagate_limit = 100
epsilon = 1e-4

image = Vec3f.field(shape=resolution)
rays = Ray.field()
hits = HitRecord.field()
ti.root.bitmasked(ti.ijk, (*resolution, 1)).bitmasked(ti.k, batch).place(rays)
ti.root.bitmasked(ti.ijk, (*resolution, 1)).bitmasked(ti.k, batch).place(hits)

@ti.func
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
    F0 = material.color
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
    return (new_dir + k * material.roughness * s).normalized()


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
    return (new_dir + k * material.roughness * s).normalized()


@ti.data_oriented
class Camera:
    def __init__(self, resolution, fov=60, focal_length=1, aperture=0):
        self.resolution = resolution
        self.fov = float(fov)
        self.focal_length = float(focal_length)
        self.aperture = float(aperture)
        self.position = Vec3f(0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

    def set_position(self, position):
        self.position = position
        
    def set_direction(self, yaw, pitch, roll=0):
        self.yaw = float(yaw) * np.pi / 180
        self.pitch = float(pitch) * np.pi / 180
        self.roll = float(roll) * np.pi / 180

    def set_fov(self, fov):
        self.fov = float(fov)

    def set_len(self, focal_length=1, aperture=0):
        self.focal_length = float(focal_length)
        self.aperture = float(aperture)

    def look_at(self, target, roll=0):
        dir = (target - self.position).normalized()
        self.yaw = ti.atan2(-dir[0], -dir[2])
        self.pitch = ti.asin(dir[1])
        self.roll = float(roll) * np.pi / 180
        
    @ti.kernel
    def get_rays(self, rays: ti.template()):
        width = self.resolution[0]
        height = self.resolution[1]

        trans = rotate(self.yaw, self.pitch, self.roll)
        ratio = height / width
        view_width = 2 * ti.tan(self.fov * np.pi / 180)
        view_height = view_width * ratio
        direction = trans @ Vec3f([0.0, 0.0, -1.0])
        width_axis = trans @ Vec3f([1.0, 0.0, 0.0])
        height_axis = trans @ Vec3f([0.0, 1.0, 0.0])
        
        for i, j, k in ti.ndrange(width, height, rays.shape[2]):
            target = self.focal_length * (direction + ((i + ti.random(ti.f32)) / width - 0.5) * view_width * width_axis + ((j + ti.random(ti.f32)) / height - 0.5) * view_height * height_axis)
            sample = sample_in_disk()
            origin = self.aperture / 2.0 * (sample[0] * width_axis + sample[1] * height_axis)
            rays[i, j, k].ro = self.position + origin
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

    def build(self, objects, max_depth=8):
        self.max_depth = max_depth
        low = Vec3f([INF, INF, INF])
        high = Vec3f([-INF, -INF, -INF])
        for object in objects:
            low = ti.min(low, object.center - object.radius)
            high = ti.max(high, object.center + object.radius)
        self.tree_nodes.append(BVHNode(left=-1, right=-1, aabb=AABB(low=low, high=high), data=0))
        self.tree_leaves.append(objects)

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
        stack = ti.Vector.zero(ti.f32, self.max_depth + 1, ti.i32)
        stack_p = 0
        stack[stack_p] = 0
        while stack_p >= 0:
            if self.tree_nodes_field[stack[stack_p]].data >= 0:   
                record = object_list_hit(self.tree_leaves_field, self.tree_leaves_field_cut[stack[stack_p]], self.tree_leaves_field_cut[stack[stack_p] + 1], ray)
                if record.t > epsilon and (res.t < 0 or record.t < res.t): res = record
                stack_p -= 1
            elif self.tree_nodes_field[self.tree_nodes_field[stack[stack_p]].left].data == -1:
                stack_p += 1
                stack[stack_p] = self.tree_nodes_field[stack[stack_p]].left
            elif self.tree_nodes_field[self.tree_nodes_field[stack[stack_p]].right].data == -1:
                stack_p += 1
                stack[stack_p] = self.tree_nodes_field[stack[stack_p]].right
            else:
                self.tree_nodes_field[stack[stack_p]].data = -2
                stack_p -= 1
        return res


@ti.data_oriented
class World:
    def __init__(self, objects=[]):
        self.objects = objects
        self.bvh = BVHTree()

    def add(self, object):
        self.objects.append(object)

    def build(self):
        self.bvh.build(self.objects, 0)

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
    
    return t1 > t0


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
        if record.t < epsilon and object.material.transparency:
            record.t = (-b + sqrt_discriminant) / (2.0 * a)
        record.point = ray.ro + record.t * ray.rd
        record.normal = (record.point - object.center).normalized()
        record.material = object.material
    return record


@ti.func
def object_list_hit(objects, start, end, ray):
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
def background_color(ray):
    t = 0.5*(ray.rd[1] + 1.0)
    color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.kernel
def propagate_once(rays: ti.template(), hits: ti.template()):
    for i, j, k in rays:
        record = world.hit(rays[i, j, k])
        if record.t >= 0:
            hits[i, j, k] = record
        else:
            image[i, j] += background_color(rays[i, j, k]) * rays[i, j, k].l / spp


@ti.kernel
def gen_secondary_rays(rays: ti.template(), hits: ti.template()):
    for i, j, k in hits:
        if hits[i, j, k].material.metalness:
            F0 = cal_reflectivity_metal(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
            rays[i, j, k].rd = sample_reflect(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
            rays[i, j, k].l = rays[i, j, k].l * F0
        else:
            F0 = cal_reflectivity_dielectirc(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
            if ti.random(ti.f32) > F0:
                if hits[i, j, k].material.transparency:
                    rays[i, j, k].rd = sample_refract(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
                    rays[i, j, k].l = rays[i, j, k].l * hits[i, j, k].material.color * (1 - hits[i, j, k].material.absorptivity)
                else:
                    rays[i, j, k].rd = sample_diffuse(hits[i, j, k].normal)
                    rays[i, j, k].l = rays[i, j, k].l * hits[i, j, k].material.color * (1 - hits[i, j, k].material.absorptivity)
            else:
                rays[i, j, k].rd = sample_reflect(rays[i, j, k].rd, hits[i, j, k].normal, hits[i, j, k].material)
                rays[i, j, k].l = rays[i, j, k].l
        rays[i, j, k].ro = hits[i, j, k].point + rays[i, j, k].rd * 2 * epsilon


@ti.kernel
def gamma_correction():
    for i, j in image:
        image[i, j] = image[i, j]**(1/2.2)


def render():
    global rays, hits
    image.fill(0)
    for _ in range(spp // batch):
        camera.get_rays(rays)
        for i in trange(propagate_limit):
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


def random_scene(size=11):
    world = World()

    ground = Sphere(center=Vec3f([0,-10000,0]), radius=10000, material=Material(color=Vec3f([1, 1, 1]), roughness=1, metalness=0, ior=1.5, absorptivity=0.5, transparency=0))
    world.add(ground)

    for a in range(-size, size):
        for b in range(-size, size):
            choose_mat = np.random.rand()
            center = Vec3f([a + 0.9 * np.random.rand(), 0.2, b + 0.9 * np.random.rand()])

            if (center - Vec3f([4, 0.2, 0])).norm() > 0.9:
                color = Vec3f([np.random.rand(), np.random.rand(), np.random.rand()])
                if choose_mat < 0.8:
                    # diffuse
                    sphere = Sphere(center=center, radius=0.2, material=Material(color=color, roughness=1, metalness=0, ior=1.5, absorptivity=0, transparency=0))
                    world.add(sphere)
                elif choose_mat < 0.95:
                    # metal
                    sphere = Sphere(center=center, radius=0.2, material=Material(color=0.5+0.5*color, roughness=0.5*np.random.rand(), metalness=1, ior=0, absorptivity=0, transparency=0))
                    world.add(sphere)
                else:
                    # glass
                    sphere = Sphere(center=center, radius=0.2, material=Material(color=0.75+0.25*color, roughness=0.2*np.random.rand(), metalness=0, ior=1.5, absorptivity=0, transparency=1))
                    world.add(sphere)

    sphere = Sphere(center=Vec3f([0, 1, 0]), radius=1.0, material=Material(color=Vec3f([1, 1, 1]), roughness=0, metalness=0, ior=1.5, absorptivity=0, transparency=1))
    world.add(sphere)
    sphere = Sphere(center=Vec3f([-4, 1, 0]), radius=1.0, material=Material(color=Vec3f([0.4, 0.2, 0.1]), roughness=1, metalness=0, ior=1.5, absorptivity=0, transparency=0))
    world.add(sphere)
    sphere = Sphere(center=Vec3f([4, 1, 0]), radius=1.0, material=Material(color=Vec3f([0.7, 0.6, 0.5]), roughness=0, metalness=1, ior=0, absorptivity=0, transparency=0))
    world.add(sphere)

    world.build()

    return world


camera = Camera(resolution)
camera.set_position(Vec3f([13, 2, 3]))
camera.look_at(Vec3f([0, 0, 0]))
camera.set_fov(20)
camera.set_len(10, 0.1)
world = random_scene(5)
gui = ti.GUI('SDF Path Tracer', resolution)
for i in range(50000):
    render()
    camera.set_position(Vec3f([13 + i, 2, 3]))
    gui.set_image(image)
    gui.show()
