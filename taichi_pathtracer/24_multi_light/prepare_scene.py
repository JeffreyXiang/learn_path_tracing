import trimesh
import taichi as ti
from dtypes import Vec3f
from world import World
from bvh import BVHSplitMode


ti.init(arch=ti.cpu)


world = World(texture_atlas_size=(4096, 8192), max_mat_num=1024)
world.load_scene(trimesh.load('assets/models/diorama_of_cyberpunk_city.glb'))
world.build_texture_atlas()
world.build_BVH(split_mode=BVHSplitMode.SAH)
world.save('assets/worlds/diorama_of_cyberpunk_city.npz')
