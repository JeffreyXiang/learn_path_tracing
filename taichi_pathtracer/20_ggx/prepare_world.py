import trimesh
import taichi as ti
from world import World
from bvh import BVHSplitMode
from material import Material

ti.init(arch=ti.cpu)

world = World(texture_atlas_size=(4096, 4096), max_mat_num=8)
world.load_scene(trimesh.load('assets/models/DamagedHelmet.glb'))
world.build_BVH(split_mode=BVHSplitMode.SAH)
world.build_texture_atlas()

world.save('assets/worlds/DamagedHelmet.npz')
