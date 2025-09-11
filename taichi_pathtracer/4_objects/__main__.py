import time
import taichi as ti
from dtypes import Vec3f, Ray
from camera import Camera
from world import World, Sphere


ti.init(arch=ti.gpu)

resolution = (1280, 720)

image = Vec3f.field(shape=resolution)
rays = Ray.field(shape=resolution)


@ti.func
def ray_color(ray, object: ti.template()):
    color = Vec3f(0.0)
    hit = object.hit(ray)
    if hit.t >= 0:
        color = 0.5 * (hit.normal + 1)
    else:
        t = 0.5*(ray.rd[1] + 1.0)
        color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.kernel
def shader(world: ti.template(), rays: ti.template()):
    for i, j in image:
        image[i, j] = ray_color(rays[i, j], world)
        

camera = Camera(resolution)
camera.set_direction(0, 0)
camera.set_position(Vec3f([0, 0, 3]))
camera.get_rays(rays)

sphere = Sphere(Vec3f([0.0,0.0,0.0]), 0.5)
ground = Sphere(Vec3f([0,-100.5,0.0]), 100)
world = World([sphere, ground])

start_time = time.time()
shader(world, rays)
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/4_objects.png')
