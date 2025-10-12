import time
import taichi as ti
from dtypes import Vec3f
from camera import Camera
from world import World
from postprocessing import ACES_tonemapping, gamma_correction
from environment import UniformEnvironment
from path_integrator import PathIntegrator, RandomWalkPathIntegrator, NextEventEstimationPathIntegrator


ti.init(arch=ti.gpu)

resolution = (1024, 1024)
spp = 128
batch = 32
path_integrator_rw = RandomWalkPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
)
path_integrator_nee = NextEventEstimationPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
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
camera.set_position(Vec3f([2.78, 2.73, -8]))
camera.look_at(Vec3f([2.78, 2.73, 0]))
camera.set_fov(39.3)
camera.prepare_render()

env = UniformEnvironment(Vec3f(0.0))
world = World(env=env)
world.load('assets/worlds/cornell_box.npz')
path_integrator_nee.prepare(world)

# Render the final image
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_rw)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/21_nee_rw.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_nee)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/21_nee_nee.png')
