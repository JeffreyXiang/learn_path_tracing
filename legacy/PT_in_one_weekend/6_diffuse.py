import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f, l=Vec3f)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, t=ti.f32, albedo=Vec3f)

resolution = (400, 225)
spp = 8192
propagate_limit = 100

image = Vec3f.field(shape=resolution)
image.fill(0)
rays = Ray.field()
rays_next = Ray.field()
ti.root.bitmasked(ti.ij, resolution).place(rays)
ti.root.bitmasked(ti.ij, resolution).place(rays_next)


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
def sample_at_sphere():
    z = 1 - 2 * ti.random(ti.f32)
    r = ti.sqrt(1 - z**2)
    theta = 2 * np.pi * ti.random(ti.f32)
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    return Vec3f([x, y, z])


@ti.func
def sample_diffuse(normal):
    s = sample_at_sphere()
    return (normal + s).normalized()


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
        self.fov =fov
        
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
        
        for i, j in ti.ndrange(width, height):
            rays[i, j].ro = [x, y, z]
            rays[i, j].rd = (direction + ((i + ti.random(ti.f32)) / width - 0.5) * width_axis + ((j + ti.random(ti.f32)) / height - 0.5) * height_axis).normalized()
            rays[i, j].l = Vec3f([1.0, 1.0, 1.0])


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
    def __init__(self, center, radius, color):
        self.center = center
        self.radius = radius
        self.color = color

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
            record.albedo = self.color
        return record


@ti.func
def backbround_color(ray):
    t = 0.5*(ray.rd[1] + 1.0)
    color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.kernel
def propagate_once(rays: ti.template(), rays_next: ti.template()):
    for i, j in rays:
        color = Vec3f(0.0)
        record = world.hit(rays[i, j])
        if record.t >= 0:
            rays_next[i, j].ro = record.point
            rays_next[i, j].rd = sample_diffuse(record.normal)
            rays_next[i, j].l = 0.5 * rays[i, j].l * record.albedo
        else:
            color = backbround_color(rays[i, j])
        image[i, j] += color * rays[i, j].l / spp


@ti.kernel
def post_processing():
    for i, j in image:
        c = image[i, j]
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


def render():
    global rays, rays_next
    for _ in range(spp):
        camera.get_rays(rays)
        for i in range(propagate_limit):
            rays_next.snode.parent().deactivate_all()
            propagate_once(rays, rays_next)
            rays, rays_next = rays_next, rays
    gamma_correction()


camera = Camera(resolution)
camera.set_direction(0, 0)
sphere = Sphere(Vec3f([0.0,0.0,-1.0]), 0.5, Vec3f([0.5, 0.5, 1]))
ground = Sphere(Vec3f([0,-100.5,-1]), 100, Vec3f([1, 1, 1]))
world = World([sphere, ground])
import time
start_time = time.time()
render()
print("time elapsed: {:.2f}s".format(time.time() - start_time))
ti.imwrite(image, '6_diffuse.png')
