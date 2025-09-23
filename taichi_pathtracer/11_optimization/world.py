import taichi as ti
from dtypes import SurfaceInteraction
from primitives import Sphere, SphereHitRecord



@ti.data_oriented
class World:
    def __init__(self, spheres=[]):
        self.capacity = max(len(spheres), 16)
        self.size = len(spheres)
        self.spheres = Sphere.field(shape=(self.capacity,))
        for i in range(self.size):
            self.spheres[i] = spheres[i]
    
    def add(self, sphere):
        if self.size >= self.capacity:
            self.capacity *= 2
            new_spheres = Sphere.field(shape=(self.capacity,))
            for i in range(self.size):
                new_spheres[i] = self.spheres[i]
            self.spheres = new_spheres
        self.spheres[self.size] = sphere
        self.size += 1

    @ti.func
    def hit(self, ray):
        sphere_hit_id = -1
        hit_t = -1.0
        sphere_hit_record = SphereHitRecord(0.0)
        si = SurfaceInteraction()
        for i in range(self.size):
            record = self.spheres[i].hit(ray)
            if record.t >= 1e-4 and (hit_t < 0 or record.t < hit_t):
                sphere_hit_id = i
                hit_t = record.t
                sphere_hit_record = record
        hit = sphere_hit_id >= 0
        if hit:
            self.spheres[sphere_hit_id].get_surface_interaction(si, ray, sphere_hit_record)
        if ray.rd.dot(si.normal) > 0:
            si.normal = -si.normal
            si.ior = 1 / si.ior
        return hit, si
