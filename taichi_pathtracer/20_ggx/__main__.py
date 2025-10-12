import time
import cv2
import taichi as ti
from dtypes import Vec3f
from camera import Camera
from world import World
from postprocessing import ACES_tonemapping, gamma_correction
from environment import ImageEnvironment
from path_integrator import PathIntegrator, RandomWalkPathIntegrator


ti.init(arch=ti.gpu)

resolution = (3840, 2160)
spp = 65536
batch = 256
path_integrator_is = RandomWalkPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=True,
)
path_integrator_nis = RandomWalkPathIntegrator(
    propagate_limit=32,
    BSDF_importance_sampling=False,
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
camera.set_position(Vec3f([-4, -0.05, 2.8]))
camera.look_at(Vec3f([0, -0.05, -0.2]))
camera.set_fov(40)
camera.set_len(4.5, 0.05)
camera.prepare_render()

env_map = cv2.cvtColor(cv2.imread('assets/textures/cayley_interior_2k.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) * 2
env = ImageEnvironment(env_map)
world = World(env=env)
world.load('assets/worlds/DamagedHelmet.npz')

# Render the final image
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_is)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/20_ggx_is.png')

image.fill(0.0)
ti.sync()
start_time = time.time()
render(world, camera, path_integrator_nis)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/20_ggx_nis.png')
