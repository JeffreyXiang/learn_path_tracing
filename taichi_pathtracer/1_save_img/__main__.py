import time
import taichi as ti

ti.init(arch=ti.gpu)

resolution = (256, 256)
image = ti.Vector.field(n=3, dtype=ti.f32, shape=resolution)

@ti.kernel
def shader():
    for i, j in image:        
        image[i, j] = ti.Vector([i / resolution[0], j / resolution[1], 0.0])

start_time = time.time()
shader()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/1_save_img.png')
