import math
import taichi as ti
from dtypes import Vec2f, Vec3f, Mat3f


@ti.func
def rotate(yaw, pitch, roll=0):
    yaw = ti.math.radians(yaw)
    pitch = ti.math.radians(pitch)
    roll = ti.math.radians(roll)
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


@ti.func
def sample_in_disk():
    r = ti.sqrt(ti.random(ti.f32))
    theta = 2 * ti.math.pi * ti.random(ti.f32)
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    return Vec2f([x, y])


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
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.roll = float(roll)

    def set_fov(self, fov):
        self.fov =fov

    def set_len(self, focal_length=1, aperture=0):
        self.focal_length = float(focal_length)
        self.aperture = float(aperture)

    def look_at(self, target, roll=0):
        dir = (target - self.position).normalized()
        self.yaw = math.degrees(math.atan2(-dir[0], -dir[2]))
        self.pitch = math.degrees(math.asin(dir[1]))
        self.roll = float(roll)

    @ti.kernel
    def get_rays(self, rays: ti.template()):
        width = self.resolution[0]
        height = self.resolution[1]
        x = self.position[0]
        y = self.position[1]
        z = self.position[2]

        trans = rotate(self.yaw, self.pitch, self.roll)
        ratio = height / width
        view_width = 2 * ti.tan(ti.math.radians(self.fov) / 2)
        view_height = view_width * ratio
        direction = trans @ Vec3f([0.0, 0.0, -1.0])
        width_axis = trans @ Vec3f([1.0, 0.0, 0.0])
        height_axis = trans @ Vec3f([0.0, 1.0, 0.0])
        
        for i, j in rays:
            target = self.focal_length * (direction + ((i + ti.random(ti.f32)) / width - 0.5) * view_width * width_axis + ((j + ti.random(ti.f32)) / height - 0.5) * view_height * height_axis)
            sample = sample_in_disk()
            origin = self.aperture / 2.0 * (sample[0] * width_axis + sample[1] * height_axis)
            rays[i, j].ro = self.position + origin
            rays[i, j].rd = (target - origin).normalized()
            rays[i, j].l = Vec3f([1.0, 1.0, 1.0])
            