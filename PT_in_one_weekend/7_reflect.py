import os
import numpy as np
import taichi as ti

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
ti.init(device_memory_GB=8, use_unified_memory=False, arch=ti.gpu)

Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f, l=Vec3f)
Material = ti.types.struct(color=Vec3f, roughness=ti.f32, metalness=ti.i32, ior=ti.f32, absorptivity=ti.f32)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, t=ti.f32, material=Material)

resolution = (1280, 720)
spp = 128
batch = 32
propagate_limit = 100

image = Vec3f.field(shape=resolution)
image.fill(0)
rays = Ray.field()
rays_next = Ray.field()
ti.root.bitmasked(ti.ijk, (*resolution, batch)).place(rays)
ti.root.bitmasked(ti.ijk, (*resolution, batch)).place(rays_next)


@ti.func
def rotate(yaw, pitch, roll=0):
    yaw = yaw * np.pi / 180
    pitch = pitch * np.pi / 180
    roll = roll * np.pi / 180
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
def cal_reflectivity_metal(dir: ti.template(), normal: ti.template(), material: ti.template()):
    F0 = material.color
    F = F0 + (1 - F0) * (1 + (normal.dot(dir)))**5
    return F


@ti.func
def cal_reflectivity_dielectirc(dir: ti.template(), normal: ti.template(), material: ti.template()):
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
def sample_diffuse(normal: ti.template()):
    s = sample_at_sphere()
    return (normal + s).normalized()


@ti.func
def sample_reflect(dir: ti.template(), normal: ti.template(), material: ti.template()):
    s = sample_in_sphere()
    k = -dir.dot(normal)
    new_dir = dir + 2 * k * normal
    return (new_dir + k * material.roughness * s).normalized()


@ti.data_oriented
class Camera:
    def __init__(self, resolution, fov=60):
        self.resolution = resolution
        self.fov = float(fov)
        self.position = Vec3f(0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

    def set_position(self, position):
        self.position = position
        
    def set_direction(self, yaw, pitch, roll=0):
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)

    def set_fov(self, fov):
        self.fov = float(fov)
        
    @staticmethod
    @ti.kernel
    def get_rays(self, rays: ti.template()):
        width = self.resolution[0]
        height = self.resolution[1]
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]

        trans = rotate(self.yaw, self.pitch, self.roll)
        ratio = height / width
        view_width = 2 * ti.tan(self.fov * np.pi / 180)
        view_height = view_width * ratio
        direction = trans @ Vec3f([0.0, 0.0, -1.0])
        width_axis = trans @ Vec3f([view_width, 0.0, 0.0])
        height_axis = trans @ Vec3f([0.0, view_height, 0.0])
        
        for i, j, k in ti.ndrange(width, height, rays.shape[2]):
            rays[i, j, k].ro = [x, y, z]
            rays[i, j, k].rd = (direction + ((i + ti.random(ti.f32)) / width - 0.5) * width_axis + ((j + ti.random(ti.f32)) / height - 0.5) * height_axis).normalized()
            rays[i, j, k].l = Vec3f([1.0, 1.0, 1.0])


@ti.data_oriented
class World:
    def __init__(self, objects=[]):
        self.objects = objects

    @ti.func
    def hit(self, ray):
        res = HitRecord(0.0)
        res.t = -1
        for i in ti.static(range(len(self.objects))):
            record = self.objects[i].hit(ray)
            if record.t > 1e-3 and (res.t < 0 or record.t < res.t): res = record
        return res


@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    @ti.func
    def hit(self, ray):
        oc = ray.ro - self.center
        a = 1
        b = 2.0 * oc.dot(ray.rd)
        c = oc.dot(oc) - self.radius**2
        discriminant = b**2 - 4 * a * c
        record = HitRecord(0.0)
        record.t = -1
        if discriminant >= 0:
            record.t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            record.point = ray.ro + record.t * ray.rd
            record.normal = (record.point - self.center).normalized()
            record.material = self.material
        return record


@ti.func
def backbround_color(ray):
    t = 0.5*(ray.rd[1] + 1.0)
    color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.kernel
def propagate_once(rays: ti.template(), rays_next: ti.template()):
    for i, j, k in rays:
        color = Vec3f(0.0)
        record = world.hit(rays[i, j, k])
        if record.t >= 0:
            rays_next[i, j, k].ro = record.point
            if record.material.metalness:
                F0 = cal_reflectivity_metal(rays[i, j, k].rd, record.normal, record.material)
                rays_next[i, j, k].rd = sample_reflect(rays[i, j, k].rd, record.normal, record.material)
                rays_next[i, j, k].l = rays[i, j, k].l * F0
            else:
                F0 = cal_reflectivity_dielectirc(rays[i, j, k].rd, record.normal, record.material)
                if ti.random(ti.f32) > F0:
                    rays_next[i, j, k].rd = sample_diffuse(record.normal)
                    rays_next[i, j, k].l = rays[i, j, k].l * record.material.color * record.material.absorptivity
                else:
                    rays_next[i, j, k].rd = sample_reflect(rays[i, j, k].rd, record.normal, record.material)
                    rays_next[i, j, k].l = rays[i, j, k].l
        else:
            color = backbround_color(rays[i, j, k])
        image[i, j] += color * rays[i, j, k].l / spp


@ti.kernel
def gamma_correction():
    for i, j in image:
        image[i, j] = image[i, j]**(1/2.2)


def render():
    global rays, rays_next
    for _ in range(spp // batch):
        camera.get_rays(rays)
        for i in range(propagate_limit):
            rays_next.snode.parent().deactivate_all()
            propagate_once(rays, rays_next)
            rays, rays_next = rays_next, rays
    gamma_correction()


camera = Camera(resolution)
camera.set_direction(0, 0)
camera.set_fov(45)
camera.set_position(Vec3f([0, -0.5, 4]))
sphere1 = Sphere(Vec3f([0.0,0.0,0.0]), 0.5, material=Material(color=Vec3f([0.5, 0.5, 1]), roughness=1, metalness=0, ior=1.5, absorptivity=0.5))
sphere2 = Sphere(Vec3f([-1.0,0.0,0.0]), 0.5, material=Material(color=Vec3f([0.5, 1, 0.5]), roughness=0, metalness=1, ior=1.5, absorptivity=0.5))
sphere3 = Sphere(Vec3f([1.0,0.0,0.0]), 0.5, material=Material(color=Vec3f([1, 0.5, 0.5]), roughness=0.25, metalness=1, ior=1.5, absorptivity=0.5))
ground = Sphere(Vec3f([0,-10000.5,0.0]), 10000, material=Material(color=Vec3f([0.5, 0.5, 0.5]), roughness=0.2, metalness=1, ior=1.5, absorptivity=0.5))
world = World([sphere1, sphere2, sphere3, ground])
import time
start_time = time.time()
render()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
ti.imwrite(image, '7_reflect.png')
