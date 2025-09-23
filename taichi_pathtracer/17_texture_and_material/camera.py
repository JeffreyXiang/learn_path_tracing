import math
import taichi as ti
from dtypes import Vec2f, Vec3f, Mat3f, Ray


def rotate(yaw, pitch, roll=0):
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    yaw_trans = Mat3f([
        [ math.cos(yaw), 0, math.sin(yaw)],
        [             0, 1,             0],
        [-math.sin(yaw), 0, math.cos(yaw)],
    ])
    pitch_trans = Mat3f([
        [1,               0,                0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch),  math.cos(pitch)],
    ])
    roll_trans = Mat3f([
        [math.cos(roll), -math.sin(roll), 0],
        [math.sin(roll),  math.cos(roll), 0],
        [             0,               0, 1],
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

    def prepare_render(self):
        trans = rotate(self.yaw, self.pitch, self.roll)
        ratio = self.resolution[1] / self.resolution[0]
        self.view_width = 2 * math.tan(math.radians(self.fov) / 2)
        self.view_height = self.view_width * ratio
        self.direction = trans @ Vec3f([0.0, 0.0, -1.0])
        self.width_axis = trans @ Vec3f([1.0, 0.0, 0.0])
        self.height_axis = trans @ Vec3f([0.0, 1.0, 0.0])

    @ti.func
    def get_ray(self, i, j):
        target = self.focal_length * (
            self.direction + \
            ((i + ti.random(ti.f32)) / self.resolution[0] - 0.5) * self.view_width * self.width_axis + \
            ((j + ti.random(ti.f32)) / self.resolution[1] - 0.5) * self.view_height * self.height_axis
        )
        sample = sample_in_disk()
        origin = self.aperture / 2.0 * (sample[0] * self.width_axis + sample[1] * self.height_axis)
        return Ray(
            ro=self.position + origin,
            rd=(target - origin).normalized(),
            l=Vec3f([1.0, 1.0, 1.0])
        )
            