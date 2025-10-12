import taichi as ti
from dtypes import Vec2f, Vec3f, AABB


LightHitRecord = ti.types.struct(t=ti.f32)


class LightType:
    SPHERE = 0
    SQUARE = 1
    DISK = 2
    TRIANGLE = 3


def _lum(c):
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z


@ti.dataclass
class Light:
    type: ti.u8         # 0: sphere, 1: square, 2: disk, 3: triangle
    position: Vec3f 
    x: Vec3f            # local x axis, only for square and disk, if sphere, norm of x is radius
    y: Vec3f            # local y axis, only for square and disk
    radiance: Vec3f    # RGB color
    material_id: ti.i32
    
    def AABB(self):
        if self.type == LightType.SPHERE:   # sphere
            radius = self.x.norm()
            return AABB(low=self.position - radius, high=self.position + radius)
        elif self.type == LightType.TRIANGLE: # triangle
            v0 = self.position
            v1 = self.position + self.x
            v2 = self.position + self.y
            low = ti.math.min(ti.math.min(v0, v1), v2)
            high = ti.math.max(ti.math.max(v0, v1), v2)
            return AABB(low=low, high=high)
        else:   # square or disk
            corners = [self.position + self.x + self.y,
                       self.position + self.x - self.y,
                       self.position - self.x + self.y,
                       self.position - self.x - self.y]
            low = ti.math.min(ti.math.min(corners[0], corners[1]), ti.math.min(corners[2], corners[3]))
            high = ti.math.max(ti.math.max(corners[0], corners[1]), ti.math.max(corners[2], corners[3]))
            return AABB(low=low, high=high)
        
    def power(self):
        lum = _lum(self.radiance)
        if self.type == LightType.SPHERE:       # sphere
            return lum * 4.0 * ti.math.pi**2 * self.x.norm()**2
        else:                                   # square or disk or triangle
            area = self.x.cross(self.y).norm()
            if self.type == LightType.SQUARE:   # square
                return 4.0 * area * ti.math.pi * lum
            elif self.type == LightType.DISK:   # disk
                return ti.math.pi * area * ti.math.pi * lum
            else:                               # triangle
                return 0.5 * area * ti.math.pi * lum
            
    @ti.func    
    def area(self):
        area = 0.0
        if self.type == LightType.SPHERE:       # sphere
            area = 4.0 * ti.math.pi * self.x.norm()**2
        else:                                   # square or disk or triangle
            area = self.x.cross(self.y).norm()
            if self.type == LightType.SQUARE:   # square
                area *= 4.0
            elif self.type == LightType.DISK:   # disk
                area *= ti.math.pi
            else:                               # triangle
                area *= 0.5
        return area
            
    @ti.func
    def sample(self):
        u1 = ti.random()
        u2 = ti.random()
        normal = Vec3f(0.0)
        pos = Vec3f(0.0)
        if self.type == LightType.SPHERE:       # sphere
            radius = self.x.norm()
            z = u1 * 2.0 - 1.0
            xy = ti.math.sqrt(1 - z**2)
            phi = u2 * 2.0 * ti.math.pi
            x = xy * ti.cos(phi)
            y = xy * ti.sin(phi)
            normal = Vec3f(x, y, z)
            pos = self.position + normal * radius
        else:                                   # square or disk or triangle
            normal = self.x.cross(self.y).normalized()
            if self.type == LightType.SQUARE:   # square
                pos = self.position + self.x * (2 * u1 - 1) + self.y * (2 * u2 - 1)
            elif self.type == LightType.DISK:   # disk
                r = ti.math.sqrt(u1)
                theta = 2 * ti.math.pi * u2
                x = r * ti.cos(theta)
                y = r * ti.sin(theta)
                pos = self.position + x * self.x + y * self.y
            else:                               # triangle
                if u1 + u2 > 1:
                    u1 = 1 - u1
                    u2 = 1 - u2
                pos = self.position + self.x * u1 + self.y * u2
        return pos, normal

    @ti.func
    def hit(self, ray):
        record = LightHitRecord(-1.0)
        if self.type == LightType.SPHERE:       # sphere
            radius2 = self.x.dot(self.x)
            oc = ray.ro - self.position
            a = 1
            b = 2.0 * ti.math.dot(oc, ray.rd)
            c = ti.math.dot(oc, oc) - radius2
            discriminant = b**2 - 4 * a * c
            if discriminant >= 0:
                sqrt_discriminant = ti.sqrt(discriminant)
                record.t = (-b - sqrt_discriminant) / (2.0 * a)
        else:                                   # square or disk or triangle
            T = ray.ro - self.position
            p = ray.rd.cross(self.y)
            det = self.x.dot(p)
            if abs(det) > 1e-12:
                inv_det = 1.0 / det
                q = T.cross(self.x)
                t = self.y.dot(q) * inv_det
                if t > 1e-4:
                    u = T.dot(p) * inv_det
                    v = ray.rd.dot(q) * inv_det
                    if self.type == LightType.SQUARE:   # square
                        if -1 <= u <= 1 and -1 <= v <= 1:
                            record.t = t
                    elif self.type == LightType.DISK:   # disk
                        if u*u + v*v <= 1:
                            record.t = t
                    else:                               # triangle
                        if u >= 0 and v >= 0 and u + v <= 1:
                            record.t = t
        return record

    @ti.func
    def get_surface_interaction(self, si:ti.template(), ray, record, material_slots, texture_atlas):
        si.point = ray.ro + ray.rd * record.t
        N = Vec3f(0.0)
        T = Vec3f(0.0)
        B = Vec3f(0.0)
        uv = Vec2f(0.0)
        
        if self.type == LightType.SPHERE:   # sphere
            N = (si.point - self.position).normalized()
            T = Vec3f(0.0, 1.0, 0.0).cross(N).normalized()
            B = N.cross(T)
        else:                               # square or disk or triangle
            N = self.x.cross(self.y).normalized()
            T = self.x.normalized()
            B = self.y.normalized()
        
        si.geo_normal = N
        si.albedo, si.metallic, si.roughness, si.normal, si.ior, si.transparency = \
            material_slots.sample(texture_atlas, self.material_id, uv, T, B, N)
        if N.dot(ray.rd) < 0:
            si.emission = self.radiance
        return si
