import time
import math
import cv2
import taichi as ti
from dtypes import Vec3f
from camera import Camera
from world import World
from postprocessing import ACES_tonemapping, gamma_correction
from primitives import Triangle
from lights import Light
from bvh import BVHSplitMode
from environment import ImageEnvironment
from material import Material
from path_integrator import PathIntegrator, NextEventEstimationPathIntegrator


ti.init(arch=ti.gpu)

resolution = (1280, 720)
spp = 1024
batch = 32
path_integrator_is = NextEventEstimationPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
    nee_multi_importance_sampling=True,
    envmap_importance_sampling=True,
)
path_integrator_nis = NextEventEstimationPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
    nee_multi_importance_sampling=True,
    envmap_importance_sampling=False
)

image = Vec3f.field(shape=resolution)


@ti.kernel
def shader(world: ti.template(), camera: ti.template(), path_integrator: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            c += path_integrator.run(ray, world) / spp
        image[i, j] += c


@ti.kernel
def post_processing():
    for i, j in image:
        c = image[i, j]
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


def render(world: World, camera: Camera, path_integrator: PathIntegrator):
    shader(world, camera, path_integrator)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([8, 0, 0]))
camera.look_at(Vec3f([0, -1, 0]))
camera.set_fov(40)
camera.prepare_render()

env_map = cv2.cvtColor(cv2.imread('assets/textures/cayley_interior_2k.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) * 2
env = ImageEnvironment(env_map)
world = World(env=env)
world.load('assets/worlds/reflective_bar.npz')
path_integrator_is.prepare(world)
path_integrator_nis.prepare(world)

# Render the final image
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_is)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/23_envmap_is_is.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_nis)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/23_envmap_is_nis.png')
