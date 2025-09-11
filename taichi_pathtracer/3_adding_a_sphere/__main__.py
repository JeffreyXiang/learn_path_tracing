import time
import taichi as ti
from dtypes import Vec3f, Ray
from camera import Camera


ti.init(arch=ti.gpu)

resolution = (1280, 720)

image = Vec3f.field(shape=resolution)
rays = Ray.field(shape=resolution)


@ti.func
def hit_sphere(center, radius, ray):
    oc = ray.ro - center
    a = 1
    b = 2.0 * ti.math.dot(oc, ray.rd)
    c = ti.math.dot(oc, oc) - radius**2
    discriminant = b**2 - 4 * a * c
    res = -1.0
    if discriminant >= 0:
        res = (-b - ti.sqrt(discriminant)) / (2.0 * a)
    return res


@ti.func
def ray_color(ray):
    center = Vec3f([0, 0, -2])
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
def shader(rays: ti.template()):
    for i, j in image:
        image[i, j] = ray_color(rays[i, j])
        

camera = Camera(resolution)
camera.set_direction(0, 0)
camera.get_rays(rays)

start_time = time.time()
shader(rays)
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/3_adding_a_sphere.png')
