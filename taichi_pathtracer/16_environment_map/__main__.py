import time
import random
import imageio
import trimesh
import taichi as ti
from dtypes import Vec3f, Material
from camera import Camera
from primitives import Sphere, Triangle
from world import World
from bsdf import MetalBSDF, DielectricBSDF
from postprocessing import ACES_tonemapping, gamma_correction
from bvh import BVHSplitMode
from environment import ImageEnvironment


def random_spheres(world, size=11):
    for a in range(-size, size):
        for b in range(-size, size):
            choose_mat = random.random()
            center = Vec3f([a + 0.9 * random.random(), 0.2, b + 0.9 * random.random()])

            if (center - Vec3f([-2, 0.2, 0])).norm() > 0.9 and \
               (center - Vec3f([ 0, 0.2, 0])).norm() > 0.9 and \
               (center - Vec3f([ 2, 0.2, 0])).norm() > 0.9:
                albedo = Vec3f([random.random(), random.random(), random.random()])
                if choose_mat < 0.8:
                    # diffuse
                    sphere = Sphere(center, 0.2, material=Material(albedo=albedo, roughness=random.random(), metallic=0, ior=1.5, transparency=0))
                    world.add_sphere(sphere)
                elif choose_mat < 0.95:
                    # metal
                    sphere = Sphere(center, 0.2, material=Material(albedo=0.5+0.5*albedo, roughness=0.5*random.random(), metallic=1, ior=0, transparency=0))
                    world.add_sphere(sphere)
                else:
                    # glass
                    sphere = Sphere(center, 0.2, material=Material(albedo=0.75+0.25*albedo, roughness=0.2*random.random(), metallic=0, ior=1.5, transparency=1))
                    world.add_sphere(sphere)


ti.init(arch=ti.gpu)

resolution = (1280, 720)
spp = 8192
batch = 32
propagate_limit = 32

image = Vec3f.field(shape=resolution)


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
                c += world.env.sample(ray) * ray.l / spp
        image[i, j] += c


@ti.kernel
def post_processing():
    for i, j in image:
        c = image[i, j]
        c = ACES_tonemapping(c)
        c = gamma_correction(c, 2.2)
        image[i, j] = c


def render(world: World, camera: Camera):
    shader(world, camera)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([-0.1, 2, 10]))
camera.look_at(Vec3f([-0.1, 0.9, 0]))
camera.set_fov(40)
camera.set_len(10, 0.2)
camera.prepare_render()

ground0 = Triangle(
    Vec3f([-50,0,50]), Vec3f([50,0,50]), Vec3f([50,0,-50]),
    Vec3f([0,1,0]), Vec3f([0,1,0]), Vec3f([0,1,0]),
    material=Material(albedo=Vec3f([0.25, 0.25, 0.25]), roughness=0.5, metallic=0, ior=1.5, transparency=0)
)
ground1 = Triangle(
    Vec3f([-50,0,50]), Vec3f([50,0,-50]), Vec3f([-50,0,-50]),
    Vec3f([0,1,0]), Vec3f([0,1,0]), Vec3f([0,1,0]),
    material=Material(albedo=Vec3f([0.25, 0.25, 0.25]), roughness=0.5, metallic=0, ior=1.5, transparency=0)
)
# Read mesh
mesh = trimesh.load_mesh('assets/models/bunny_3k.ply')
bounds = mesh.bounds
mesh.vertices -= bounds.mean(axis=0)
mesh.vertices /= (bounds[1] - bounds[0]).max()
mesh.vertices *= 2
mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()
triangles = [
    # center, plastic
    Triangle(
        mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]],
        mesh.vertex_normals[mesh.faces[i][0]], mesh.vertex_normals[mesh.faces[i][1]], mesh.vertex_normals[mesh.faces[i][2]],
        material=Material(albedo=Vec3f([0.8, 0.8, 0.8]), roughness=0.5, metallic=0, ior=1.5, transparency=0)
    )
    for i in range(len(mesh.faces))
]
mesh.vertices[:, 0] -= 2
triangles += [
    # left, glass
    Triangle(
        mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]],
        mesh.vertex_normals[mesh.faces[i][0]], mesh.vertex_normals[mesh.faces[i][1]], mesh.vertex_normals[mesh.faces[i][2]],
        material=Material(albedo=Vec3f([0.8, 0.8, 0.8]), roughness=0, metallic=0, ior=1.5, transparency=1)
    )
    for i in range(len(mesh.faces))
]
mesh.vertices[:, 0] += 4
triangles += [
    # right, copper
    Triangle(
        mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]], 
        mesh.vertex_normals[mesh.faces[i][0]], mesh.vertex_normals[mesh.faces[i][1]], mesh.vertex_normals[mesh.faces[i][2]],
        material=Material(albedo=Vec3f([0.955, 0.638, 0.538]), roughness=0.2, metallic=1, ior=1.5, transparency=0)
    )
    for i in range(len(mesh.faces))
]
env_map = imageio.imread('assets/textures/cayley_interior_2k.exr') / 50.0
env = ImageEnvironment(env_map)
world = World(spheres=[], triangles=[ground0, ground1] + triangles, env=env)
random_spheres(world)
world.build_BVH(split_mode=BVHSplitMode.SAH)

# Render the final image
ti.sync()
start_time = time.time()
render(world, camera)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/16_environment_map.png')