import time
import taichi as ti
from dtypes import Vec3f, Ray
from camera import Camera


ti.init(arch=ti.gpu)

resolution = (1280, 720)

image = Vec3f.field(shape=resolution)
rays = Ray.field(shape=resolution)


@ti.func
def ray_color(ray):
    t = 0.5*(ray.rd[1] + 1.0)
    return (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])


@ti.kernel
def shader(rays: ti.template()):
    for i, j in image:
        image[i, j] = ray_color(rays[i, j])


camera = Camera(resolution)
camera.set_direction(0, 30, 0)
camera.get_rays(rays)

start_time = time.time()
shader(rays)
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/2_camera_and_ray.png')
