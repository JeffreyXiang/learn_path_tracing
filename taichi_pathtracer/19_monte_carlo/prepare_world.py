import taichi as ti
from dtypes import Vec3f
from world import World
from primitives import Triangle
from lights import Light
from bvh import BVHSplitMode
from material import Material

ti.init(arch=ti.cpu)

world = World(texture_atlas_size=(1, 1), max_mat_num=8)

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
green_mat = world.add_material(Material(
    baseColorFactor=Vec3f(0.05, 0.4, 0.02),
    baseColorTexture=-1,
    metallicFactor=0.0,
    roughnessFactor=1.0,
    metallicRoughnessTexture=-1,
    normalTexture=-1,
    transmissionFactor=0.0,
    ior=1.0,
))
red_mat = world.add_material(Material(
    baseColorFactor=Vec3f(0.4, 0.035, 0.01),
    baseColorTexture=-1,
    metallicFactor=0.0,
    roughnessFactor=1.0,
    metallicRoughnessTexture=-1,
    normalTexture=-1,
    transmissionFactor=0.0,
    ior=1.0,
))

world.triangles = [
    # Floor (white)
    Triangle(v0=Vec3f([5.528, 0.0, 0.0]), v1=Vec3f([0.0, 0.0, 0.0]), v2=Vec3f([0.0, 0.0, 5.592]),
             material_id=white_mat),
    Triangle(v0=Vec3f([5.528, 0.0, 0.0]), v1=Vec3f([0.0, 0.0, 5.592]), v2=Vec3f([5.496, 0.0, 5.592]),
             material_id=white_mat),
    
    # Ceiling (white)
    Triangle(v0=Vec3f([5.56, 5.488, 0.0]), v1=Vec3f([5.56, 5.488, 5.592]), v2=Vec3f([0.0, 5.488, 5.592]),
             material_id=white_mat),
    Triangle(v0=Vec3f([5.56, 5.488, 0.0]), v1=Vec3f([0.0, 5.488, 5.592]), v2=Vec3f([0.0, 5.488, 0.0]),
             material_id=white_mat),

    # Back wall (white)
    Triangle(v0=Vec3f([5.496, 0.0, 5.592]), v1=Vec3f([0.0, 0.0, 5.592]), v2=Vec3f([0.0, 5.488, 5.592]),
             material_id=white_mat),
    Triangle(v0=Vec3f([5.496, 0.0, 5.592]), v1=Vec3f([0.0, 5.488, 5.592]), v2=Vec3f([5.56, 5.488, 5.592]),
             material_id=white_mat),

    # Right wall (green)
    Triangle(v0=Vec3f([0.0, 0.0, 5.592]), v1=Vec3f([0.0, 0.0, 0.0]), v2=Vec3f([0.0, 5.488, 0.0]),
             material_id=green_mat),
    Triangle(v0=Vec3f([0.0, 0.0, 5.592]), v1=Vec3f([0.0, 5.488, 0.0]), v2=Vec3f([0.0, 5.488, 5.592]),
             material_id=green_mat),
    
    # Left wall (red)
    Triangle(v0=Vec3f([5.528, 0.0, 0.0]), v1=Vec3f([5.496, 0.0, 5.592]), v2=Vec3f([5.56, 5.488, 5.592]),
             material_id=red_mat),
    Triangle(v0=Vec3f([5.528, 0.0, 0.0]), v1=Vec3f([5.56, 5.488, 5.592]), v2=Vec3f([5.56, 5.488, 0.0]),
             material_id=red_mat),

    # Short box (white)
    Triangle(v0=Vec3f([1.3, 1.65, 0.65]), v1=Vec3f([0.82, 1.65, 2.25]), v2=Vec3f([2.4, 1.65, 2.72]),
             material_id=white_mat),
    Triangle(v0=Vec3f([1.3, 1.65, 0.65]), v1=Vec3f([2.4, 1.65, 2.72]), v2=Vec3f([2.9, 1.65, 1.14]),
             material_id=white_mat),
    Triangle(v0=Vec3f([2.9, 0.0, 1.14]), v1=Vec3f([2.9, 1.65, 1.14]), v2=Vec3f([2.4, 1.65, 2.72]),
             material_id=white_mat),
    Triangle(v0=Vec3f([2.9, 0.0, 1.14]), v1=Vec3f([2.4, 1.65, 2.72]), v2=Vec3f([2.4, 0.0, 2.72]),
             material_id=white_mat),
    Triangle(v0=Vec3f([1.3, 0.0, 0.65]), v1=Vec3f([1.3, 1.65, 0.65]), v2=Vec3f([2.9, 1.65, 1.14]),
             material_id=white_mat),
    Triangle(v0=Vec3f([1.3, 0.0, 0.65]), v1=Vec3f([2.9, 1.65, 1.14]), v2=Vec3f([2.9, 0.0, 1.14]),
             material_id=white_mat),
    Triangle(v0=Vec3f([0.82, 0.0, 2.25]), v1=Vec3f([0.82, 1.65, 2.25]), v2=Vec3f([1.3, 1.65, 0.65]),
             material_id=white_mat),
    Triangle(v0=Vec3f([0.82, 0.0, 2.25]), v1=Vec3f([1.3, 1.65, 0.65]), v2=Vec3f([1.3, 0.0, 0.65]),
             material_id=white_mat),
    Triangle(v0=Vec3f([2.4, 0.0, 2.72]), v1=Vec3f([2.4, 1.65, 2.72]), v2=Vec3f([0.82, 1.65, 2.25]),
             material_id=white_mat),
    Triangle(v0=Vec3f([2.4, 0.0, 2.72]), v1=Vec3f([0.82, 1.65, 2.25]), v2=Vec3f([0.82, 0.0, 2.25]),
             material_id=white_mat),
    
    # Tall box (white)
    Triangle(v0=Vec3f([4.23, 0.0, 2.47]), v1=Vec3f([4.23, 3.3, 2.47]), v2=Vec3f([4.72, 3.3, 4.06]),
             material_id=white_mat),
    Triangle(v0=Vec3f([4.23, 0.0, 2.47]), v1=Vec3f([4.72, 3.3, 4.06]), v2=Vec3f([4.72, 0.0, 4.06]),
             material_id=white_mat),
    Triangle(v0=Vec3f([4.72, 0.0, 4.06]), v1=Vec3f([4.72, 3.3, 4.06]), v2=Vec3f([3.14, 3.3, 4.56]),
             material_id=white_mat),
    Triangle(v0=Vec3f([4.72, 0.0, 4.06]), v1=Vec3f([3.14, 3.3, 4.56]), v2=Vec3f([3.14, 0.0, 4.56]),
             material_id=white_mat),
    Triangle(v0=Vec3f([3.14, 0.0, 4.56]), v1=Vec3f([3.14, 3.3, 4.56]), v2=Vec3f([2.65, 3.3, 2.96]),
             material_id=white_mat),
    Triangle(v0=Vec3f([3.14, 0.0, 4.56]), v1=Vec3f([2.65, 3.3, 2.96]), v2=Vec3f([2.65, 0.0, 2.96]),
             material_id=white_mat),
    Triangle(v0=Vec3f([2.65, 0.0, 2.96]), v1=Vec3f([2.65, 3.3, 2.96]), v2=Vec3f([4.23, 3.3, 2.47]),
             material_id=white_mat),
    Triangle(v0=Vec3f([2.65, 0.0, 2.96]), v1=Vec3f([4.23, 3.3, 2.47]), v2=Vec3f([4.23, 0.0, 2.47]),
             material_id=white_mat),
]
for tri in world.triangles:
    normal = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).normalized()
    tri.n0 = normal
    tri.n1 = normal
    tri.n2 = normal
    
world.lights = [
    Light(
        type=1,
        position=Vec3f([2.78, 5.48, 2.745]),
        x=Vec3f([0.65, 0.0, 0.0]),
        y=Vec3f([0.0, 0.0, 0.525]),
        radiance=Vec3f([100.0, 70.0, 30.0]),
        material_id=white_mat,
    )
]

world.build_BVH(split_mode=BVHSplitMode.SAH)

world.save('assets/worlds/cornell_box.npz')
