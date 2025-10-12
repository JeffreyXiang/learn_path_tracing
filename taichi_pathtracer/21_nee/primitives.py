import taichi as ti
from dtypes import Vec2f, Vec3f, AABB


SphereHitRecord = ti.types.struct(t=ti.f32)
TriangleHitRecord = ti.types.struct(t=ti.f32, w0=ti.f32, w1=ti.f32)


@ti.dataclass
class Sphere:
    center: Vec3f
    radius: ti.f32
    material_id: ti.i32

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
            if record.t < 1e-4:
                record.t = (-b + sqrt_discriminant) / (2.0 * a)
        return record

    @ti.func
    def get_surface_interaction(self, si:ti.template(), ray, record, material_slots, texture_atlas):
        point = ray.ro + record.t * ray.rd
        N = (point - self.center).normalized()
        T = Vec3f(0.0, 1.0, 0.0).cross(N).normalized()
        B = N.cross(T)
        uv = Vec2f(
            ti.math.atan2(N[0], N[2]) / ti.math.pi + 1,
            ti.math.asin(N[1]) / ti.math.pi + 0.5,
        )
        si.point = point
        si.geo_normal = N
        si.albedo, si.metallic, si.roughness, si.normal, si.ior, si.transparency = \
            material_slots.sample(texture_atlas, self.material_id, uv, T, B, N)
        return si


@ti.dataclass
class Triangle:
    v0: Vec3f
    v1: Vec3f
    v2: Vec3f
    n0: Vec3f
    n1: Vec3f
    n2: Vec3f
    t0: Vec2f
    t1: Vec2f
    t2: Vec2f
    material_id: ti.i32

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
                    record.w0 = w0
                    record.w1 = w1
        return record

    @ti.func
    def get_surface_interaction(self, si:ti.template(), ray, record, material_slots, texture_atlas):
        uv = record.w0 * self.t0 + record.w1 * self.t1 + (1 - record.w0 - record.w1) * self.t2
        # calculate TBN
        N = (record.w0 * self.n0 + record.w1 * self.n1 + (1 - record.w0 - record.w1) * self.n2).normalized()
        T = Vec3f(0.0)
        B = Vec3f(0.0)
        dp1 = self.v1 - self.v0
        dp2 = self.v2 - self.v0
        duv1 = self.t1 - self.t0
        duv2 = self.t2 - self.t0
        f = duv1[0] * duv2[1] - duv2[0] * duv1[1]
        if abs(f) > 1e-8:
            f = 1.0 / f
            T = (f * (duv2[1] * dp1 - duv1[1] * dp2)).normalized()
            B = (f * (-duv2[0] * dp1 + duv1[0] * dp2)).normalized()
        # fill in the surface interaction
        si.point = ray.ro + ray.rd * record.t
        si.geo_normal = (self.v1 - self.v0).cross(self.v2 - self.v0).normalized()
        si.albedo, si.metallic, si.roughness, si.normal, si.ior, si.transparency = \
            material_slots.sample(texture_atlas, self.material_id, uv, T, B, N)
        # if shading normal and geometric normal are not in the same direction, clip shading normal
        if si.normal.dot(ray.rd) * si.geo_normal.dot(ray.rd) < 0:
            si.normal = (si.normal - 1.001 * ray.rd.dot(si.normal) * ray.rd).normalized()
        return si
