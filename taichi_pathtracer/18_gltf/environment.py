import numpy as np
import taichi as ti
from dtypes import Vec3f


@ti.data_oriented
class GradientEnvironment:
    def __init__(self, color1, color2):
        self.color1 = color1
        self.color2 = color2

    @ti.func
    def sample(self, ray):
        t = 0.5*(ray.rd[1] + 1.0)
        color = (1.0-t)*self.color2 + t*self.color1
        return color


@ti.data_oriented
class ImageEnvironment:
    def __init__(self, img):
        self.width = img.shape[1]
        self.height = img.shape[0]
        self.map = Vec3f.field(shape=(self.width, self.height))
        self.map.from_numpy(np.flip(img.transpose(1, 0, 2), 1))

    @ti.func
    def sample(self, ray):
        u = (ti.math.atan2(ray.rd[2], ray.rd[0]) / (2.0 * ti.math.pi) + 0.5) * self.width
        v = (ti.math.asin(ti.math.clamp(ray.rd[1], -1, 1)) / ti.math.pi + 0.5) * self.height
        l = ti.math.floor(u - 0.5) + 0.5
        r = l + 1.0
        b = ti.math.floor(v - 0.5) + 0.5
        t = b + 1.0
        w1 = (r - u) * (t - v)
        w2 = (u - l) * (t - v)
        w3 = (r - u) * (v - b)
        w4 = (u - l) * (v - b)
        l = (l + self.width) % self.width
        r = (r + self.width) % self.width
        b = ti.math.clamp(b, 0, self.height - 1)
        t = ti.math.clamp(t, 0, self.height - 1)
        c1 = self.map[int(l), int(b)]
        c2 = self.map[int(r), int(b)]
        c3 = self.map[int(l), int(t)]
        c4 = self.map[int(r), int(t)]
        color = ti.math.max(0.0, w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4)
        return color
