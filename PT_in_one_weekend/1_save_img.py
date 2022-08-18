import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

image = ti.Vector.field(n=3, dtype=ti.f32, shape=(256, 256))

@ti.kernel
def render():
    for i, j in image:
        t = j / 255.0
        image[i, j] = t * ti.Vector([0.3, 0.5, 0.9]) + (1 - t) * ti.Vector([0.8, 0.9, 1.0])


render()
ti.imwrite(image, '1_save_img.png')
