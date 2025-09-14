import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f)

resolution = (400, 225)
image = Vec3f.field(shape=resolution)


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
    def __init__(self, resolution, fov=60):
        self.resolution = resolution
        self.fov = float(fov)
        self.position = Vec3f(0)
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.rays = Ray.field(shape=resolution)

    def set_position(self, position):
        self.position = position
        
    def set_direction(self, yaw, pitch, roll=0):
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)

    def set_fov(self, fov):
        self.fov =fov
        
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
        
        for i, j in self.rays:
            self.rays[i, j].ro = [x, y, z]
            self.rays[i, j].rd = (direction + (i / (width - 1) - 0.5) * width_axis + (j / (height - 1) - 0.5) * height_axis).normalized()


@ti.func
def hit_sphere(center, radius, ray):
    oc = ray.ro - center
    a = 1
    b = 2.0 * ti.dot(oc, ray.rd)
    c = ti.dot(oc, oc) - radius**2
    discriminant = b**2 - 4 * a * c
    res = -1.0
    if discriminant >= 0:
        res = (-b - ti.sqrt(discriminant)) / (2.0 * a)
    return res


@ti.func
def ray_color(ray):
    center = Vec3f([0, 0, -1])
    radius = 0.5
    color = Vec3f(0.0)
    t = hit_sphere(center, radius, ray)
    if t > 0:
        normal = (ray.ro + t * ray.rd - center).normalized()
        color = 0.5 * (normal + 1)
    else:
        t = 0.5*(ray.rd[1] + 1.0)
        color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.kernel
def render(rays: ti.template()):
    for i, j in image:
        image[i, j] = ray_color(rays[i, j])
        

camera = Camera(resolution)
camera.set_direction(0, 0)
camera.get_rays()
render(camera.rays)
ti.imwrite(image, '3_adding_a_sphere.png')
