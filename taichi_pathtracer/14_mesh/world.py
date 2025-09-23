import taichi as ti
from dtypes import SurfaceInteraction, BVHTraverseStatistics
from primitives import Sphere, SphereHitRecord, Triangle, TriangleHitRecord
from bvh import BVH, BVHSplitMode


@ti.data_oriented
class World:
    def __init__(self, spheres=[], triangles=[]):
        self.spheres = spheres
        self.spheres_BVH = None
        self.triangles = triangles
        self.triangles_BVH = None
    
    def add_sphere(self, sphere):
        self.spheres.append(sphere)

    def add_triangle(self, triangle):
        self.triangles.append(triangle)

    def build_BVH(self, max_depth=None, split_mode=BVHSplitMode.SAH):
        if self.spheres_BVH is None:
            self.spheres_BVH = BVH(Sphere, SphereHitRecord, len(self.spheres))
        self.spheres_BVH.build(self.spheres, max_depth, split_mode)
        if self.triangles_BVH is None:
            self.triangles_BVH = BVH(Triangle, TriangleHitRecord, len(self.triangles))
        self.triangles_BVH.build(self.triangles, max_depth, split_mode)

    @ti.func
    def hit(self, ray):
        sphere_hit_id, sphere_hit_record, sphere_vis = self.spheres_BVH.hit(ray)
        triangle_hit_id, triangle_hit_record, triangle_vis = self.triangles_BVH.hit(ray)

        hit_primitive_type = -1
        hit_t = 1e10
        si = SurfaceInteraction()
        vis = BVHTraverseStatistics()
        if sphere_hit_id >= 0 and sphere_hit_record.t < hit_t:
            hit_primitive_type = 0
            hit_t = sphere_hit_record.t
            vis.nfe_aabb += sphere_vis.nfe_aabb
            vis.nfe_primitive += sphere_vis.nfe_primitive
        if triangle_hit_id >= 0 and triangle_hit_record.t < hit_t:
            hit_primitive_type = 1
            hit_t = triangle_hit_record.t
            vis.nfe_aabb += triangle_vis.nfe_aabb
            vis.nfe_primitive += triangle_vis.nfe_primitive

        if hit_primitive_type == 0:
            self.spheres_BVH.primitives[sphere_hit_id].get_surface_interaction(si, ray, sphere_hit_record)
        elif hit_primitive_type == 1:
            self.triangles_BVH.primitives[triangle_hit_id].get_surface_interaction(si, ray, triangle_hit_record)

        if ray.rd.dot(si.normal) > 0:
            si.normal = -si.normal
            si.ior = 1 / si.ior

        return sphere_hit_id >= 0 or triangle_hit_id >= 0, si, vis
