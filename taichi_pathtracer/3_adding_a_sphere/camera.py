import taichi as ti
from dtypes import Vec3f, Mat3f


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
        width_axis = trans @ Vec3f([view_width, 0.0, 0.0])
        height_axis = trans @ Vec3f([0.0, view_height, 0.0])
        
        for i, j in rays:
            rays[i, j].ro = [x, y, z]
            rays[i, j].rd = (direction + (i / (width - 1) - 0.5) * width_axis + (j / (height - 1) - 0.5) * height_axis).normalized()
            