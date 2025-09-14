import time
from tqdm import trange
import taichi as ti
from dtypes import Vec3f, Ray, Material
from camera import Camera
from world import World, Sphere
from bsdf import MetalBSDF, DielectricBSDF
from postprocessing import ACES_tonemapping, gamma_correction


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
def post_processing():
    for i, j in image:
        c = image[i, j]
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


def render(world: World, camera: Camera):
    for _ in trange(spp):
        camera.get_rays(rays)
        shader(world, rays)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([3, 0.5, 2]))
camera.look_at(Vec3f([0.0,0.35,0.0]))
camera.set_len(focal_length=camera.position.norm(), aperture=0.2)

sphere1 = Sphere(Vec3f([0.0,0.0,0.0]), 0.5, material=Material(albedo=Vec3f([0.25, 0.25, 0.5]), roughness=0.5, metallic=0, ior=1.5))
sphere2 = Sphere(Vec3f([-1.0,0.0,0.0]), 0.5, material=Material(albedo=Vec3f([0.25, 0.5, 0.25]), roughness=0, metallic=1, ior=1.5))
sphere3 = Sphere(Vec3f([1.0,0.0,0.0]), 0.5, material=Material(albedo=Vec3f([0.5, 0.25, 0.25]), roughness=0.5, metallic=1, ior=1.5))
sphere4 = Sphere(Vec3f([-0.5,0.866,0]), 0.5, material=Material(albedo=Vec3f([1, 1, 1]), roughness=0, metallic=0, ior=1.5, transparency=1))
sphere5 = Sphere(Vec3f([0.5,0.866,0]), 0.5, material=Material(albedo=Vec3f([0.5, 1, 0.5]), roughness=0.5, metallic=0, ior=1.5, transparency=1))
ground = Sphere(Vec3f([0,-10000.5,0.0]), 10000, material=Material(albedo=Vec3f([0.25, 0.25, 0.25]), roughness=0.5, metallic=0, ior=1.5))
world = World([sphere1, sphere2, sphere3, sphere4, sphere5, ground])

start_time = time.time()
render(world, camera)
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/9_dof.png')
