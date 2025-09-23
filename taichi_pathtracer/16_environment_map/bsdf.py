import taichi as ti
from dtypes import Vec3f


@ti.func
def _sample_at_sphere():
    z = 1 - 2 * ti.random(ti.f32)
    r = ti.sqrt(1 - z**2)
    theta = 2 * ti.math.pi * ti.random(ti.f32)
    x = r * ti.cos(theta)
    y = r * ti.sin(theta)
    return Vec3f([x, y, z])


@ti.func
def _sample_lambertian(normal, geo_normal):
    s = _sample_at_sphere()
    r = normal + s
    if r.dot(geo_normal) < 0:
        r = r - 2 * r.dot(geo_normal) * geo_normal
    return r.normalized()


@ti.func
def _slerp(a, b, t):
    omega = ti.acos(ti.math.clamp(a.dot(b), -1, 1))
    so = ti.sin(omega)
    o = (1 - t) * a + t * b if so < 1e-6 else \
        (ti.sin((1 - t) * omega) / so) * a + (ti.sin(t * omega) / so) * b
    return o.normalized()


@ti.func
def _sample_normal(dir, normal, geo_normal, roughness):
    s = _sample_lambertian(normal, geo_normal)
    k = -dir.dot(normal)
    r = dir + 2 * k * normal
    if r.dot(geo_normal) < 0:
        r = r - 2 * r.dot(geo_normal) * geo_normal
    r = _slerp(r, s, roughness*roughness)
    n = (r - dir).normalized()
    return n


@ti.func
def _reflect(dir, normal):
    k = -dir.dot(normal)
    r = dir + 2 * k * normal
    return r


@ti.func
def _refract(dir, normal, ior):
    k = dir.dot(normal)
    r_out_perp = (dir - k * normal) / ior
    r_out_perp_len2 = r_out_perp.dot(r_out_perp)
    r = Vec3f(0)
    if r_out_perp_len2 > 1:
        r = _reflect(dir, normal)
    else:
        k = ti.sqrt(1.0 - r_out_perp_len2)
        r_out_parallel = -k * normal
        r = r_out_perp + r_out_parallel
    return r


class MetalBSDF:
    @staticmethod
    @ti.func
    def cal_fresnel(dir, normal, albedo):
        F0 = albedo
        cos_theta = max(0.0, normal.dot(-dir))
        return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5

    @staticmethod
    @ti.func
    def sample(ray: ti.template(), si: ti.template()):
        n = _sample_normal(ray.rd, si.normal, si.geo_normal, si.roughness)
        F = MetalBSDF.cal_fresnel(ray.rd, n, si.albedo)
        ray.l *= F
        ray.ro = si.point
        ray.rd = _reflect(ray.rd, n)


class DielectricBSDF:
    @staticmethod
    @ti.func
    def cal_fresnel(dir, normal, ior):
        F0 = ((ior - 1) / (ior + 1))**2
        cos_theta = max(0.0, normal.dot(-dir))
        return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5
    
    @staticmethod
    @ti.func
    def sample(ray: ti.template(), si: ti.template()):
        n = _sample_normal(ray.rd, si.normal, si.geo_normal, si.roughness)
        F = DielectricBSDF.cal_fresnel(ray.rd, n, si.ior)
        ray.ro = si.point
        if ti.random() > F:
            ray.l *= si.albedo
            if si.transparency:
                ray.rd = _refract(ray.rd, n, si.ior)
            else:
                ray.rd = _sample_lambertian(si.normal, si.geo_normal)
        else:
            ray.rd = _reflect(ray.rd, n)
