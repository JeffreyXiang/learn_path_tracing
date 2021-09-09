import os
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

image = ti.Vector.field(n=3, dtype=ti.f32, shape=(512, 512))

@ti.kernel
def render():
    for i, j in image:
        image[i, j] = ti.Vector([i / 512, j / 512, 0.25])


render()
ti.imwrite(image, '1_save_img.png')
