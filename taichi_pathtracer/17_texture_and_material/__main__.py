import time
import random
import imageio
import numpy as np
import trimesh
import taichi as ti
from dtypes import Vec2f, Vec3f
from camera import Camera
from primitives import Sphere, Triangle
from world import World
from bsdf import PrincipledBSDF
from postprocessing import ACES_tonemapping, gamma_correction
from bvh import BVHSplitMode
from environment import ImageEnvironment
from material import Material


def random_spheres(world, size=11):
    for a in range(-size, size):
        for b in range(-size, size):
            choose_mat = random.random()
            center = Vec3f([a + 0.9 * random.random(), 0.2, b + 0.9 * random.random()])

            if (center - Vec3f([-2, 0.2, 0])).norm() > 0.9 and \
               (center - Vec3f([ 0, 0.2, 0])).norm() > 0.9 and \
               (center - Vec3f([ 2, 0.2, 0])).norm() > 0.9:
                albedo = Vec3f([random.random(), random.random(), random.random()])
                if choose_mat < 0.9:
                    # texture
                    tex_name = random.choice(texture_names)
                    mat_id = world.add_material(Material(
                        baseColorFactor=0.75+0.25*albedo,
                        baseColorTexture=textures[tex_name]['base_color'],
                        metallicFactor=1.0,
                        roughnessFactor=1.0,
                        metallicRoughnessTexture=textures[tex_name]['metallic_roughness'],
                        normalTexture=textures[tex_name]['normal'],
                        transmissionFactor=0.0,
                        ior=1.5,
                    )) 
                    sphere = Sphere(center, 0.2, material_id=mat_id)
                    world.add_sphere(sphere)
                else:
                    # glass
                    mat_id = world.add_material(Material(
                        baseColorFactor=0.75+0.25*albedo,
                        baseColorTexture=-1,
                        metallicFactor=0.0,
                        roughnessFactor=0.2*random.random(),
                        metallicRoughnessTexture=-1,
                        normalTexture=-1,
                        transmissionFactor=1.0,
                        ior=1.5,
                    )) 
                    sphere = Sphere(center, 0.2, material_id=mat_id)
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
            c += si.albedo / spp
        image[i, j] += c


@ti.kernel
def shader_m_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            c += Vec3f(si.metallic) / spp
        image[i, j] += c


@ti.kernel
def shader_r_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            c += Vec3f(si.roughness) / spp
        image[i, j] += c
    

@ti.kernel
def shader_n_vis(world: ti.template(), camera: ti.template()):
    for i, j, k in ti.ndrange(resolution[0], resolution[1], spp//batch):
        c = Vec3f(0.0)
        for b in range(batch):
            ray = camera.get_ray(i, j)
            hit, si, vis = world.hit(ray)
            c += (si.normal * 0.5 + 0.5) / spp
        image[i, j] += c


def render(world: World, camera: Camera):
    shader(world, camera)
    post_processing()


camera = Camera(resolution)
camera.set_position(Vec3f([0, 2, 10]))
camera.look_at(Vec3f([0, 0.9, 0]))
camera.set_fov(40)
camera.set_len(10, 0.2)
camera.prepare_render()

env_map = imageio.imread('assets/textures/cayley_interior_2k.exr') / 50.0
env = ImageEnvironment(env_map)
world = World(env=env, texture_atlas_size=(12288, 12288), max_mat_num=1024)

texture_names = [
    'antique_veneer1',
    'bamboo-wood-semigloss',
    'granite-gray-white',
    'nylon-tent-fabric',
    'patchy_cement1',
    'pitted-rusted-metal1',
    'rustediron2',
    'sandyground1',
    'soft-blanket',
    'stringy_marble',
]
textures = {}
for name in texture_names:
    bc_img = imageio.imread(f'assets/textures/{name}_albedo.png')[..., :3]
    m_img = imageio.imread(f'assets/textures/{name}_metallic.png')
    r_img = imageio.imread(f'assets/textures/{name}_roughness.png')
    if m_img.ndim == 3: m_img = m_img[..., 0]
    if r_img.ndim == 3: r_img = r_img[..., 0]
    mr_img = np.stack([np.zeros_like(m_img), r_img, m_img], axis=-1)
    n_img = imageio.imread(f'assets/textures/{name}_normal.png')[..., :3]
    bc_tex_id = world.add_texture(bc_img)
    mr_tex_id = world.add_texture(mr_img)
    n_tex_id = world.add_texture(n_img)
    textures[name] = {
        'base_color': bc_tex_id,
        'metallic_roughness': mr_tex_id,
        'normal': n_tex_id,
    }
world.build_texture_atlas()

ground_mat_id = world.add_material(Material(
    baseColorFactor=Vec3f([0.25, 0.25, 0.25]),
    baseColorTexture=-1,
    metallicFactor=0.0,
    roughnessFactor=0.5,
    metallicRoughnessTexture=-1,
    normalTexture=-1,
    transmissionFactor=0.0,
    ior=1.5,
))
ground0 = Triangle(
    Vec3f([-50,0,50]), Vec3f([50,0,50]), Vec3f([50,0,-50]),
    Vec3f([0,1,0]), Vec3f([0,1,0]), Vec3f([0,1,0]),
    Vec2f([0,0]), Vec2f([1,0]), Vec2f([1,1]),
    material_id=ground_mat_id
)
ground1 = Triangle(
    Vec3f([-50,0,50]), Vec3f([50,0,-50]), Vec3f([-50,0,-50]),
    Vec3f([0,1,0]), Vec3f([0,1,0]), Vec3f([0,1,0]),
    Vec2f([0,0]), Vec2f([1,1]), Vec2f([0,1]),
    material_id=ground_mat_id
)
world.add_triangle(ground0)
world.add_triangle(ground1)
# Read mesh
mesh = trimesh.load_mesh('assets/models/bunny_3k.ply')
bounds = mesh.bounds
mesh.vertices -= bounds.mean(axis=0)
mesh.vertices /= (bounds[1] - bounds[0]).max()
mesh.vertices *= 2
mesh.vertices[:, 1] -= mesh.vertices[:, 1].min()
# center, plastic
mat_id = world.add_material(Material(
    baseColorFactor=Vec3f([0.8, 0.8, 0.8]),
    baseColorTexture=textures['stringy_marble']['base_color'],
    metallicFactor=1.0,
    roughnessFactor=1.0,
    metallicRoughnessTexture=textures['stringy_marble']['metallic_roughness'],
    normalTexture=textures['stringy_marble']['normal'],
    transmissionFactor=0.0,
    ior=1.5,
))
triangles = [
    Triangle(
        mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]],
        mesh.vertex_normals[mesh.faces[i][0]], mesh.vertex_normals[mesh.faces[i][1]], mesh.vertex_normals[mesh.faces[i][2]],
        mesh.vertices[mesh.faces[i][0]][:2], mesh.vertices[mesh.faces[i][1]][:2], mesh.vertices[mesh.faces[i][2]][:2],
        material_id=mat_id
    )
    for i in range(len(mesh.faces))
]
# left, glass
mesh.vertices[:, 0] -= 2
mat_id = world.add_material(Material(
    baseColorFactor=Vec3f([0.8, 0.8, 0.8]),
    baseColorTexture=-1,
    metallicFactor=0.0,
    roughnessFactor=0.0,
    metallicRoughnessTexture=-1,
    normalTexture=-1,
    transmissionFactor=1.0,
    ior=1.5,
))
triangles += [
    Triangle(
        mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]],
        mesh.vertex_normals[mesh.faces[i][0]], mesh.vertex_normals[mesh.faces[i][1]], mesh.vertex_normals[mesh.faces[i][2]],
        mesh.vertices[mesh.faces[i][0]][:2], mesh.vertices[mesh.faces[i][1]][:2], mesh.vertices[mesh.faces[i][2]][:2],
        material_id=mat_id
    )
    for i in range(len(mesh.faces))
]
# right, metal
mesh.vertices[:, 0] += 4
mat_id = world.add_material(Material(
    baseColorFactor=Vec3f([1.0, 1.0, 1.0]),
    baseColorTexture=textures['rustediron2']['base_color'],
    metallicFactor=1.0,
    roughnessFactor=1.0,
    metallicRoughnessTexture=textures['rustediron2']['metallic_roughness'],
    normalTexture=textures['rustediron2']['normal'],
    transmissionFactor=0.0,
    ior=1.5,
))
triangles += [
    Triangle(
        mesh.vertices[mesh.faces[i][0]], mesh.vertices[mesh.faces[i][1]], mesh.vertices[mesh.faces[i][2]], 
        mesh.vertex_normals[mesh.faces[i][0]], mesh.vertex_normals[mesh.faces[i][1]], mesh.vertex_normals[mesh.faces[i][2]],
        mesh.vertices[mesh.faces[i][0]][:2], mesh.vertices[mesh.faces[i][1]][:2], mesh.vertices[mesh.faces[i][2]][:2],
        material_id=mat_id
    )
    for i in range(len(mesh.faces))
]
world.triangles += triangles
random_spheres(world)
world.build_BVH(split_mode=BVHSplitMode.SAH)

ti.tools.imwrite(ti.tools.imresize(world.texture_atlas.atlas, 1024), 'outputs/17_texture_and_material_texture_atlas.png')


# Visualize texture
image.fill(0)
shader_bc_vis(world, camera)
ti.tools.imwrite(image, 'outputs/17_texture_and_material_basecolor.png')

image.fill(0)
shader_m_vis(world, camera)
ti.tools.imwrite(image, 'outputs/17_texture_and_material_metallic.png')

image.fill(0)
shader_r_vis(world, camera)
ti.tools.imwrite(image, 'outputs/17_texture_and_material_roughness.png')

image.fill(0)
shader_n_vis(world, camera)
ti.tools.imwrite(image, 'outputs/17_texture_and_material_normal.png')

# Render the final image
image.fill(0)
ti.sync()
start_time = time.time()
render(world, camera)
ti.sync()
print(f"Time elapsed: {time.time() - start_time:.2f}s")

ti.tools.imwrite(image, 'outputs/17_texture_and_material.png')
