import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, t=ti.f32)

resolution = (400, 225)
image = Vec3f.field(shape=resolution)
image.fill(0)


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
        [             0,             0, 1],
    ])
    return yaw_trans @ pitch_trans @ roll_trans


@ti.data_oriented
class Camera:
    def __init__(self, resolution, fov=60, spp=100):
        self.resolution = resolution
        self.fov = float(fov)
        self.spp = spp
        self.position = Vec3f(0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.rays = Ray.field(shape=(*resolution, spp))

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
    def get_rays(self):
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
        
        for i, j, k in self.rays:
            self.rays[i, j, k].ro = [x, y, z]
            self.rays[i, j, k].rd = (direction + ((i + ti.random(ti.f32)) / width - 0.5) * width_axis + ((j + ti.random(ti.f32)) / height - 0.5) * height_axis).normalized()


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
            if record.t >= 0 and (res.t < 0 or record.t < res.t): res = record
        return res


@ti.data_oriented
class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

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
        return record


@ti.func
def ray_color(ray, object: ti.template()):
    color = Vec3f(0.0)
    record = object.hit(ray)
    if record.t >= 0:
        color = 0.5 * (record.normal + 1)
    else:
        t = 0.5*(ray.rd[1] + 1.0)
        color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.kernel
def render(rays: ti.template()):
    sphere = Sphere(Vec3f([0.0,0.0,-1.0]), 0.5)
    ground = Sphere(Vec3f([0,-100.5,-1]), 100)
    world = World([sphere, ground])
    for i, j, k in rays:
        image[i, j] += ray_color(rays[i, j, k], world) / rays.shape[2]
        

camera = Camera(resolution, spp=100)
camera.set_direction(0, 0)
camera.get_rays()
render(camera.rays)
ti.imwrite(image, '5_anti_aliasing.png')
