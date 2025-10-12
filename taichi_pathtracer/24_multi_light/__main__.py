import time
import trimesh
import taichi as ti
from dtypes import Vec3f, PrimitiveType
from camera import Camera
from world import World
from postprocessing import ACES_tonemapping, gamma_correction
from bvh import BVHSplitMode
from environment import UniformEnvironment
from material import Material
from path_integrator import PathIntegrator, RandomWalkPathIntegrator, NextEventEstimationPathIntegrator


ti.init(arch=ti.gpu)

resolution = (3840, 2160)
spp = 128
batch = 32
exposure = 50.0
path_integrator_rw = RandomWalkPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=False,
)
path_integrator_rw_is = RandomWalkPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
)
path_integrator_nee = NextEventEstimationPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
    nee_multi_importance_sampling=False,
    use_light_bvh=False,
)
path_integrator_nee_bvh = NextEventEstimationPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
    nee_multi_importance_sampling=False,
    use_light_bvh=True,
)
path_integrator_nee_bvh_mis = NextEventEstimationPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
    nee_multi_importance_sampling=True,
    use_light_bvh=True,
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
        c *= exposure
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


def render(world: World, camera: Camera, path_integrator: PathIntegrator):
    shader(world, camera, path_integrator)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([14.3448, 6.31466, -13.4463]))
camera.set_direction(133.7, -12.5, 0.0)
camera.set_fov(39.6)
camera.set_len(19.53, 0.5)
camera.prepare_render()

env = UniformEnvironment(Vec3f(0.0))
world = World(env=env)
world.load('assets/worlds/diorama_of_cyberpunk_city.npz')
path_integrator_rw.prepare(world)
path_integrator_rw_is.prepare(world)
path_integrator_nee.prepare(world)
path_integrator_nee_bvh.prepare(world)
path_integrator_nee_bvh_mis.prepare(world)


# Render the final image
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_rw)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/24_multi_light_rw.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_rw_is)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/24_multi_light_rw_is.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_nee)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/24_multi_light_nee.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_nee_bvh)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/24_multi_light_nee_bvh.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_nee_bvh_mis)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/24_multi_light_nee_bvh_mis.png')
