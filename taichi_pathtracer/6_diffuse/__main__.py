import time
from tqdm import trange
import taichi as ti
from dtypes import Vec3f, Ray
from camera import Camera
from world import World, Sphere
from bsdf import DiffuseBSDF


ti.init(arch=ti.gpu)

resolution = (1280, 720)
spp = 8192
propagate_limit = 32

image = Vec3f.field(shape=resolution)
rays = Ray.field(shape=resolution)


@ti.func
def backbround_color(ray):
    t = 0.5*(ray.rd[1] + 1.0)
    color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.func
def propagate_once(ray: ti.template(), world: ti.template()):
    if ray.end == 0:
        hit = world.hit(ray)
        if hit.t >= 0:
            DiffuseBSDF.sample(ray, hit)
        else:
            ray.end = ti.int8(1)


@ti.kernel
def shader(world: ti.template(), rays: ti.template()):
    for i, j in rays:
        ray = rays[i, j]
        for _ in range(propagate_limit):
            propagate_once(ray, world)
            if ray.end == 1:
                break
        if ray.end == 1:
            image[i, j] += backbround_color(ray) * ray.l / spp


@ti.kernel
def gamma_correction():
    for i, j in image:
        image[i, j] = image[i, j]**(1/2.2)


def render(world: World, camera: Camera):
    for _ in trange(spp):
        camera.get_rays(rays)
        shader(world, rays)
    gamma_correction()


camera = Camera(resolution)
camera.set_direction(0, 0)
camera.set_position(Vec3f([0, 0, 3]))

sphere = Sphere(Vec3f([0.0,0.0,0.0]), 0.5, Vec3f([0.5, 0.5, 0.5]))
ground = Sphere(Vec3f([0,-100.5,0]), 100, Vec3f([0.5, 0.5, 0.5]))
world = World([sphere, ground])

start_time = time.time()
render(world, camera)
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/6_diffuse.png')
