import time
import random
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
from dtypes import Vec3f, Material
from camera import Camera
from primitives import Sphere
from world import World
from bsdf import MetalBSDF, DielectricBSDF
from postprocessing import ACES_tonemapping, gamma_correction
from bvh import BVHSplitMode


def random_scene(num=1000):
    world = World()

    for _ in range(num):
        center = Vec3f([3 - 6 * random.random(), 0, 3 - 6 * random.random()])
        albedo = Vec3f([random.random(), random.random(), random.random()])
        sphere = Sphere(center, 0.2, material=Material(albedo=albedo, roughness=random.random(), metallic=0, ior=1.5, transparency=0))
        world.add(sphere)

    return world


ti.init(arch=ti.gpu)

resolution = (1024, 1024)
spp = 8192
batch = 32
propagate_limit = 32

image = Vec3f.field(shape=resolution)


@ti.func
def backbround_color(ray):
    t = 0.5*(ray.rd[1] + 1.0)
    color = (1.0-t)*Vec3f([1.0, 1.0, 1.0]) + t*Vec3f([0.5, 0.7, 1.0])
    return color


@ti.func
def propagate_once(ray: ti.template(), world: ti.template()):
    if ray.end == 0:
        hit, si, vis = world.hit(ray)
        if hit:
            if si.metallic == 1:
                MetalBSDF.sample(ray, si)
            else:
                DielectricBSDF.sample(ray, si)
        else:
            ray.end = ti.int8(1)


@ti.kernel
def shader(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            for _ in range(propagate_limit):
                propagate_once(ray, world)
                if ray.end == 1:
                    break
            if ray.end == 1:
                c += backbround_color(ray) * ray.l / spp
        image[i, j] += c


@ti.kernel
def post_processing():
    for i, j in image:
        c = image[i, j]
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


@ti.kernel
def shader_nfe_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            c += Vec3f([vis.nfe_aabb, vis.nfe_primitive, 0.0]) / spp
        image[i, j] += c


def render(world: World, camera: Camera):
    shader(world, camera)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([0, 100, 0]))
camera.set_direction(0, -90, 0)
camera.set_fov(4)
camera.set_len(100, 0.2)
camera.prepare_render()

world = random_scene()

nfes = {}
for split_mode in [BVHSplitMode.EQUAL, BVHSplitMode.MIDDLE, BVHSplitMode.SAH]:
    world.build_BVH(split_mode=split_mode)
    print(f"[{split_mode}] BVH Depth: {world.spheres_BVH.depth}")

    # Visualize the number of intersection tests per ray
    image.fill(0)
    shader_nfe_vis(world, camera)

    image_nfe_aabb = np.flip(image.to_numpy()[..., 0].T, axis=0)
    print(f'[{split_mode}] Average number of AABB intersection tests per ray: {image_nfe_aabb.mean()}')

    image_nfe_primitive = np.flip(image.to_numpy()[..., 1].T, axis=0)
    print(f'[{split_mode}] Average number of primitive intersection tests per ray: {image_nfe_primitive.mean()}')
    
    nfes[split_mode] = image_nfe_aabb + image_nfe_primitive

    # Render the final image
    image.fill(0)
    ti.sync()
    start_time = time.time()
    render(world, camera)
    ti.sync()
    print(f"[{split_mode}] Time elapsed: {time.time() - start_time:.2f}s")

    ti.tools.imwrite(image, f'outputs/13_better_bvh_{split_mode}.png')
    
# Visualize the number of intersection tests per ray
vmax = max(max(nfes[split_mode].max() for split_mode in nfes), 1)
plt.figure(figsize=(24, 8), constrained_layout=True)
axs = []
for i in range(3):
    axs.append(plt.subplot(1, 3, i+1))
    plt.axis("off")
    plt.imshow(nfes[list(nfes.keys())[i]], vmin=0, vmax=vmax)
plt.colorbar(shrink=1, ax=axs, pad=0.01)
plt.savefig(f"outputs/13_better_bvh_nfe.png")
plt.close()
