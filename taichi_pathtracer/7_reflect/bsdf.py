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
def _sample_in_sphere():
    r = ti.random(ti.f32)**(1.0 / 3.0)
    theta = 2 * ti.math.pi * ti.random(ti.f32)
    phi = ti.acos(ti.random(ti.f32) * 2 - 1)
    x = r * ti.cos(theta) * ti.sin(phi)
    y = r * ti.sin(theta) * ti.sin(phi)
    z = r * ti.cos(phi)
    return Vec3f([x, y, z])


@ti.func
def _sample_lambertian(normal):
    s = _sample_at_sphere()
    return (normal + s).normalized()


@ti.func
def _sample_reflect(dir, normal, roughness):
    s = _sample_in_sphere()
    k = -dir.dot(normal)
    new_dir = dir + 2 * k * normal
    return (new_dir + k * roughness * s).normalized()


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
        F = MetalBSDF.cal_fresnel(ray.rd, hit.normal, hit.material.albedo)
        ray.l *= F
        ray.ro = hit.point
        ray.rd = _sample_reflect(ray.rd, hit.normal, hit.material.roughness)


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
        F = DielectricBSDF.cal_fresnel(ray.rd, hit.normal, hit.material.ior)
        ray.ro = hit.point
        if ti.random() > F:
            ray.rd = _sample_lambertian(hit.normal)
            ray.l *= hit.material.albedo
        else:
            ray.rd = _sample_reflect(ray.rd, hit.normal, hit.material.roughness)
