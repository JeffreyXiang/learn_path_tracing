import time
import random
import imageio
import numpy as np
import trimesh
import taichi as ti
from dtypes import Vec3f
from camera import Camera
from world import World
from bsdf import PrincipledBSDF
from postprocessing import ACES_tonemapping, gamma_correction
from bvh import BVHSplitMode
from environment import ImageEnvironment
from material import Material


ti.init(arch=ti.gpu)

resolution = (3840, 2160)
spp = 8192
batch = 32
propagate_limit = 32

image = Vec3f.field(shape=resolution)


@ti.func
def propagate_once(ray: ti.template(), world: ti.template()):
    if ray.end == 0:
        hit, si, vis = world.hit(ray)
        if hit:
            PrincipledBSDF.sample(ray, si)
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
                c += world.env.sample(ray) * ray.l / spp
        image[i, j] += c


@ti.kernel
def post_processing():
    for i, j in image:
        c = image[i, j]
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


@ti.kernel
def shader_bc_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            if hit:
                c += si.albedo / spp
            else:
                c += gamma_correction(ACES_tonemapping(world.env.sample(ray)), 2.2) / spp
        image[i, j] += c


@ti.kernel
def shader_m_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            if hit:
                c += Vec3f(si.metallic) / spp
            else:
                c += gamma_correction(ACES_tonemapping(world.env.sample(ray)), 2.2) / spp
        image[i, j] += c


@ti.kernel
def shader_r_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            if hit:
                c += Vec3f(si.roughness) / spp
            else:
                c += gamma_correction(ACES_tonemapping(world.env.sample(ray)), 2.2) / spp
        image[i, j] += c
    

@ti.kernel
def shader_n_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            if hit:
                c += (si.normal * 0.5 + 0.5) / spp
            else:
                c += gamma_correction(ACES_tonemapping(world.env.sample(ray)), 2.2) / spp
        image[i, j] += c


def render(world: World, camera: Camera):
    shader(world, camera)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([-4, -0.05, 2.8]))
camera.look_at(Vec3f([0, -0.05, -0.2]))
camera.set_fov(40)
camera.set_len(4.5, 0.05)
camera.prepare_render()

env_map = imageio.imread('assets/textures/cayley_interior_2k.exr') / 50.0
env = ImageEnvironment(env_map)
world = World(env=env, texture_atlas_size=(4096, 4096), max_mat_num=1024)
world.load_gltf('assets/models/DamagedHelmet.glb')
world.build_BVH(split_mode=BVHSplitMode.SAH)
world.build_texture_atlas()
ti.tools.imwrite(ti.tools.imresize(world.texture_atlas.atlas, 1024), 'outputs/18_gltf_texture_atlas.png')

# Visualize the material properties
image.fill(0)
shader_bc_vis(world, camera)
ti.tools.imwrite(image, 'outputs/18_gltf_basecolor.png')

image.fill(0)
shader_m_vis(world, camera)
ti.tools.imwrite(image, 'outputs/18_gltf_metallic.png')

image.fill(0)
shader_r_vis(world, camera)
ti.tools.imwrite(image, 'outputs/18_gltf_roughness.png')

image.fill(0)
shader_n_vis(world, camera)
ti.tools.imwrite(image, 'outputs/18_gltf_normal.png')

# Render the final image
image.fill(0)
ti.sync()
start_time = time.time()
render(world, camera)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/18_gltf.png')
