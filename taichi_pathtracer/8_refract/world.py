import taichi as ti
from dtypes import Vec3f, Material, HitRecord


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
        res = HitRecord(0.0)
        res.t = -1
        for i in range(self.size):
            record = self.spheres[i].hit(ray)
            if record.t >= 1e-4 and (res.t < 0 or record.t < res.t): res = record
        if ray.rd.dot(res.normal) > 0:
            res.normal = -res.normal
            res.material.ior = 1 / res.material.ior
        return res


@ti.dataclass
class Sphere:
    center: Vec3f
    radius: ti.f32
    material: Material

    @ti.func
    def hit(self, ray):
        oc = ray.ro - self.center
        a = 1
        b = 2.0 * ti.math.dot(oc, ray.rd)
        c = ti.math.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4 * a * c
        record = HitRecord(0.0)
        record.t = -1
        if discriminant >= 0:
            sqrt_discriminant = ti.sqrt(discriminant)
            record.t = (-b - sqrt_discriminant) / (2.0 * a)
            if record.t < 1e-4 and self.material.transparency:
                record.t = (-b + sqrt_discriminant) / (2.0 * a)
            record.point = ray.ro + record.t * ray.rd
            record.normal = (record.point - self.center).normalized()
            record.material = self.material
        return record
