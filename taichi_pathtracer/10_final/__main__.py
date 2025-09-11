import time
import random
from tqdm import trange
import taichi as ti
from dtypes import Vec3f, Ray, Material
from camera import Camera
from world import World, Sphere
from bsdf import MetalBSDF, DielectricBSDF


def random_scene(size=8):
    world = World()

    ground = Sphere(Vec3f([0,-10000,0]), 10000, material=Material(albedo=Vec3f([0.25, 0.25, 0.25]), roughness=1, metallic=0, ior=1.5, transparency=0))
    world.add(ground)

    for a in range(-size, size):
        for b in range(-size, size):
            choose_mat = random.random()
            center = Vec3f([a + 0.9 * random.random(), 0.2, b + 0.9 * random.random()])

            if (center - Vec3f([4, 0.2, 0])).norm() > 0.9:
                albedo = Vec3f([random.random(), random.random(), random.random()])
                if choose_mat < 0.8:
                    # diffuse
                    sphere = Sphere(center, 0.2, material=Material(albedo=albedo, roughness=1, metallic=0, ior=1.5, transparency=0))
                    world.add(sphere)
                elif choose_mat < 0.95:
                    # metal
                    sphere = Sphere(center, 0.2, material=Material(albedo=0.5+0.5*albedo, roughness=0.5*random.random(), metallic=1, ior=0, transparency=0))
                    world.add(sphere)
                else:
                    # glass
                    sphere = Sphere(center, 0.2, material=Material(albedo=0.75+0.25*albedo, roughness=0.2*random.random(), metallic=0, ior=1.5, transparency=1))
                    world.add(sphere)

    sphere = Sphere(Vec3f([0, 1, 0]), 1.0, material=Material(albedo=Vec3f([1, 1, 1]), roughness=0, metallic=0, ior=1.5, transparency=1))
    world.add(sphere)
    sphere = Sphere(Vec3f([-4, 1, 0]), 1.0, material=Material(albedo=Vec3f([0.4, 0.2, 0.1]), roughness=1, metallic=0, ior=1.5, transparency=0))
    world.add(sphere)
    sphere = Sphere(Vec3f([4, 1, 0]), 1.0, material=Material(albedo=Vec3f([0.7, 0.6, 0.5]), roughness=0, metallic=1, ior=0, transparency=0))
    world.add(sphere)

    return world


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
            if hit.material.metallic == 1:
                MetalBSDF.sample(ray, hit)
            else:
                DielectricBSDF.sample(ray, hit)
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
camera.set_position(Vec3f([13, 2, 3]))
camera.look_at(Vec3f([0, 0, 0]))
camera.set_fov(40)
camera.set_len(10, 0.2)

world = random_scene()

start_time = time.time()
render(world, camera)
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/10_final.png')
