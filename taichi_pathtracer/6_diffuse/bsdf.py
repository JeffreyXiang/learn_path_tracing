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


class DiffuseBSDF:
    @staticmethod
    @ti.func
    def sample(ray: ti.template(), hit: ti.template()):
        ray.l *= hit.albedo
        ray.ro = hit.point
        ray.rd = _sample_lambertian(hit.normal)
