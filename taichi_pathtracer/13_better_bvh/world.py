import taichi as ti
from dtypes import SurfaceInteraction
from primitives import Sphere, SphereHitRecord
from bvh import BVH, BVHSplitMode


@ti.data_oriented
class World:
    def __init__(self, spheres=[]):
        self.spheres = spheres
        self.spheres_BVH = None
    
    def add(self, sphere):
        self.spheres.append(sphere)

    def build_BVH(self, max_depth=None, split_mode=BVHSplitMode.SAH):
        if self.spheres_BVH is None:
            self.spheres_BVH = BVH(Sphere, SphereHitRecord, len(self.spheres))
        self.spheres_BVH.build(self.spheres, max_depth, split_mode)

    @ti.func
    def hit(self, ray):
        si = SurfaceInteraction()
        sphere_hit_id, sphere_hit_record, vis = self.spheres_BVH.hit(ray)
        hit = sphere_hit_id >= 0
        if hit:
            self.spheres_BVH.primitives[sphere_hit_id].get_surface_interaction(si, ray, sphere_hit_record)
        if ray.rd.dot(si.normal) > 0:
            si.normal = -si.normal
            si.ior = 1 / si.ior
        return hit, si, vis
