import taichi as ti
from dtypes import Vec3f, Material, AABB


SphereHitRecord = ti.types.struct(t=ti.f32) 
TriangleNaiveHitRecord = ti.types.struct(t=ti.f32, point=Vec3f, normal=Vec3f)
TriangleMTHitRecord = ti.types.struct(t=ti.f32)


@ti.dataclass
class Sphere:
    center: Vec3f
    radius: ti.f32
    material: Material

    def AABB(self):
        return AABB(low=self.center - self.radius, high=self.center + self.radius)

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


@ti.dataclass
class TriangleNaive:
    v0: Vec3f
    v1: Vec3f
    v2: Vec3f
    material: Material

    def AABB(self):
        aabb = AABB(low=ti.math.min(self.v0, ti.math.min(self.v1, self.v2)),
                    high=ti.math.max(self.v0, ti.math.max(self.v1, self.v2)))
        return aabb

    @ti.func
    def hit(self, ray):
        record = TriangleHitRecord(0.0, 0.0, 0.0)
        record.t = -1
        edge0 = self.v1 - self.v0
        edge1 = self.v2 - self.v1
        edge2 = self.v0 - self.v2
        N = edge0.cross(edge1)
        inv_area2 = 1.0 / N.dot(N)
        denom = ray.rd.dot(N)
        if ti.abs(denom) > 1e-12:
            t = (N.dot(self.v0) - ray.ro.dot(N)) / denom
            if (t > 1e-4):
                # intersection
                P = ray.ro + ray.rd * t
                # calculate barycentric coordinates
                w0 = edge1.cross(P - self.v1).dot(N) * inv_area2
                w1 = edge2.cross(P - self.v2).dot(N) * inv_area2
                w2 = 1 - w0 - w1
                hit = (w0 > 0 and w1 > 0 and w2 > 0)
                # hit, fill in the record
                if hit:
                    record.t = t
                    record.point = P
                    record.normal = N.normalized()
        return record

    @ti.func
    def get_surface_interaction(self, si:ti.template(), ray, record):
        si.point = record.point
        si.normal = record.normal
        si.albedo = self.material.albedo
        si.metallic = self.material.metallic
        si.roughness = self.material.roughness
        si.ior = self.material.ior
        si.transparency = self.material.transparency
        return si
    

@ti.dataclass
class TriangleMT:
    v0: Vec3f
    v1: Vec3f
    v2: Vec3f
    material: Material
    
    def AABB(self):
        aabb = AABB(low=ti.math.min(self.v0, ti.math.min(self.v1, self.v2)),
                    high=ti.math.max(self.v0, ti.math.max(self.v1, self.v2)))
        return aabb

    @ti.func
    def hit(self, ray):
        # using Möller-Trumbore algorithm
        record = TriangleHitRecord(-1.0)
        e1 = self.v1 - self.v0
        e2 = self.v2 - self.v0
        T = ray.ro - self.v0
        p = ray.rd.cross(e2)
        det = e1.dot(p)
        if abs(det) > 1e-12:        
            inv_det = 1.0 / det
            q = T.cross(e1)
            t = e2.dot(q) * inv_det
            if t > 1e-4:
                # solve barycentric coordinates
                w1 = T.dot(p) * inv_det
                w2 = ray.rd.dot(q) * inv_det
                w0 = 1.0 - w1 - w2
                # hit, fill in the record
                if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0 and t > 0.0:
                    record.t = t
        return record
    
    @ti.func
    def get_surface_interaction(self, si:ti.template(), ray, record):
        si.point = ray.ro + ray.rd * record.t
        si.normal = (self.v1 - self.v0).cross(self.v2 - self.v0).normalized()
        si.albedo = self.material.albedo
        si.metallic = self.material.metallic
        si.roughness = self.material.roughness
        si.ior = self.material.ior
        si.transparency = self.material.transparency
        return si
    

TriangleHitRecord = TriangleMTHitRecord
Triangle = TriangleMT
