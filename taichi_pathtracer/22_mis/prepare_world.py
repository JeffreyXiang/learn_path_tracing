import math
import taichi as ti
from dtypes import Vec3f
from world import World
from primitives import Triangle
from lights import Light
from bvh import BVHSplitMode
from material import Material

ti.init(arch=ti.cpu)

world = World(texture_atlas_size=(1, 1), max_mat_num=8)

roughnesses = [0.0, 0.17, 0.33, 0.5]
ellipse_a2 = 20
seg = 0.6
start = -0.2

white_mat = world.add_material(Material(
    baseColorFactor=Vec3f(0.8, 0.8, 0.8),
    baseColorTexture=-1,
    metallicFactor=0.0,
    roughnessFactor=1.0,
    metallicRoughnessTexture=-1,
    normalTexture=-1,
    transmissionFactor=0.0,
    ior=1.0,
))

corner_x = -0.8
corner_y = -math.sqrt((1 - (start + len(roughnesses) * seg - 4)**2 / ellipse_a2) * (ellipse_a2 - 16))
world.triangles.extend([
    Triangle(v0=Vec3f([corner_x, corner_y, 100]), v1=Vec3f([corner_x, corner_y, -100]), v2=Vec3f([corner_x, 100, -100]),
             material_id=white_mat),
    Triangle(v0=Vec3f([corner_x, corner_y, 100]), v1=Vec3f([corner_x, 100, -100]), v2=Vec3f([corner_x, 100, 100]),
             material_id=white_mat),
    Triangle(v0=Vec3f([corner_x, corner_y, 100]), v1=Vec3f([100, corner_y, 100]), v2=Vec3f([100, corner_y, -100]),
             material_id=white_mat),
    Triangle(v0=Vec3f([corner_x, corner_y, 100]), v1=Vec3f([100, corner_y, -100]), v2=Vec3f([corner_x, corner_y, -100]),
             material_id=white_mat),
])

mats = [
    world.add_material(Material(
        baseColorFactor=Vec3f(0.3, 0.4, 0.5),
        baseColorTexture=-1,
        metallicFactor=1.0,
        roughnessFactor=r,
        metallicRoughnessTexture=-1,
        normalTexture=-1,
        transmissionFactor=0.0,
        ior=1.0,
    )) for r in [0.05, 0.1, 0.15, 0.2]
]

for i in range(len(roughnesses)):
    x0 = start + (i + 0.1) * seg
    x1 = start + (i + 0.9) * seg
    y0 = -math.sqrt((1 - (x0 - 4)**2 / ellipse_a2) * (ellipse_a2 - 16))
    y1 = -math.sqrt((1 - (x1 - 4)**2 / ellipse_a2) * (ellipse_a2 - 16))
    z0 = -2
    z1 = 2
    world.triangles.extend([
        Triangle(v0=Vec3f([x0, y0, z0]), v1=Vec3f([x1, y1, z0]), v2=Vec3f([x1, y1, z1]),
                 material_id=mats[i]),
        Triangle(v0=Vec3f([x0, y0, z0]), v1=Vec3f([x1, y1, z1]), v2=Vec3f([x0, y0, z1]),
                 material_id=mats[i]),
    ])

for tri in world.triangles:
    normal = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).normalized()
    tri.n0 = normal
    tri.n1 = normal
    tri.n2 = normal
    
light_mat = world.add_material(Material(
    baseColorFactor=Vec3f(1.0),
    baseColorTexture=-1,
    metallicFactor=0.0,
    roughnessFactor=1.0,
    metallicRoughnessTexture=-1,
    normalTexture=-1,
    transmissionFactor=0.0,
    ior=1.0,
))

world.lights = [
    Light(
        type=0,
        position=Vec3f([0, 0, z]),
        x=Vec3f([r, 0.0, 0.0]),
        radiance=Vec3f(0.5 / r**2),
        material_id=light_mat,
    )
    for z, r in [(1.4, 0.4), (0.2, 0.15), (-0.8, 0.05), (-1.6, 0.02)]
]

world.build_BVH(split_mode=BVHSplitMode.SAH)

world.save('assets/worlds/reflective_bar.npz')
