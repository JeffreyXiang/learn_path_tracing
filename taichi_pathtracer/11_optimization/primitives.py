import taichi as ti
from dtypes import Vec3f, Material


SphereHitRecord = ti.types.struct(t=ti.f32)


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
        record = SphereHitRecord(0.0)
        record.t = -1
        if discriminant >= 0:
            sqrt_discriminant = ti.sqrt(discriminant)
            record.t = (-b - sqrt_discriminant) / (2.0 * a)
            if record.t < 1e-4 and self.material.transparency:
                record.t = (-b + sqrt_discriminant) / (2.0 * a)
        return record

    @ti.func
    def get_surface_interaction(self, si:ti.template(), ray, record):
        si.point = ray.ro + record.t * ray.rd
        si.normal = (si.point - self.center).normalized()
        si.albedo = self.material.albedo
        si.metallic = self.material.metallic
        si.roughness = self.material.roughness
        si.ior = self.material.ior
        si.transparency = self.material.transparency
        return si
