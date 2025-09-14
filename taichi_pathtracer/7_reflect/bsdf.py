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
def _sample_lambertian(normal):
    s = _sample_at_sphere()
    return (normal + s).normalized()


@ti.func
def _slerp(a, b, t):
    omega = ti.acos(ti.math.clamp(a.dot(b), -1, 1))
    so = ti.sin(omega)
    o = (1 - t) * a + t * b if so < 1e-6 else \
        (ti.sin((1 - t) * omega) / so) * a + (ti.sin(t * omega) / so) * b
    return o.normalized()


@ti.func
def _sample_normal(dir, normal, roughness):
    s = _sample_lambertian(normal)
    k = -dir.dot(normal)
    r = dir + 2 * k * normal
    r = _slerp(r, s, roughness*roughness)
    n = (r - dir).normalized()
    return n


@ti.func
def _reflect(dir, normal):
    k = -dir.dot(normal)
    r = dir + 2 * k * normal
    return r


class DiffuseBSDF:
    @staticmethod
    @ti.func
    def sample(ray: ti.template(), hit: ti.template()):
        ray.l *= hit.albedo
        ray.ro = hit.point
        ray.rd = _sample_lambertian(hit.normal)


class MetalBSDF:
    @staticmethod
    @ti.func
    def cal_fresnel(dir, normal, albedo):
        F0 = albedo
        cos_theta = max(0.0, normal.dot(-dir))
        return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5

    @staticmethod
    @ti.func
    def sample(ray: ti.template(), hit: ti.template()):
        n = _sample_normal(ray.rd, hit.normal, hit.material.roughness)
        F = MetalBSDF.cal_fresnel(ray.rd, n, hit.material.albedo)
        ray.l *= F
        ray.ro = hit.point
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
    def sample(ray: ti.template(), hit: ti.template()):
        n = _sample_normal(ray.rd, hit.normal, hit.material.roughness)
        F = DielectricBSDF.cal_fresnel(ray.rd, n, hit.material.ior)
        ray.ro = hit.point
        if ti.random() > F:
            ray.l *= hit.material.albedo
            ray.rd = _sample_lambertian(hit.normal)
        else:
            ray.rd = _reflect(ray.rd, n)
