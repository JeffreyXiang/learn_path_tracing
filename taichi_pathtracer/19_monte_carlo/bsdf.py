import taichi as ti
from dtypes import Vec3f


@ti.func
def _get_tangent_space(n):
    # Frisvad’s method
    t = Vec3f(0.0)
    b = Vec3f(0.0)
    if n[2] < -0.9999999:
        t[1] = -1.0
        b[0] = -1.0
    else:
        a = 1.0 / (1.0 + n[2])
        bb = -n[0] * n[1] * a
        t[0] = 1.0 - n[0] * n[0] * a
        t[1] = bb
        t[2] = -n[0]
        b[0] = bb
        b[1] = 1 - n[1] * n[1] * a
        b[2] = -n[1]
    return t, b

@ti.func
def _sample_hemisphere_uniform(t, b, n):
    u1 = ti.random(ti.f32)
    u2 = ti.random(ti.f32)
    z = u1
    phi = 2.0 * ti.math.pi * u2
    r = ti.sqrt(1.0 - z * z)
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    local_dir = Vec3f([x, y, z])
    return t * local_dir.x + b * local_dir.y + n * local_dir.z


@ti.func
def _sample_hemisphere_cosine_weighted(t, b, n):
    u1 = ti.random(ti.f32)
    u2 = ti.random(ti.f32)
    r = ti.sqrt(u1)
    phi = 2.0 * ti.math.pi * u2
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    z = ti.sqrt(max(0.0, 1.0 - r * r))
    local_dir = Vec3f([x, y, z])
    return t * local_dir.x + b * local_dir.y + n * local_dir.z


@ti.func
def _reflect(dir, normal):
    k = -dir.dot(normal)
    r = dir + 2 * k * normal
    return r


class DiffuseBSDF:
    @staticmethod
    @ti.func
    def f(wo, wi, si: ti.template()):
        return si.albedo / ti.pi

    @staticmethod
    @ti.func
    def sample(wo, si: ti.template(), use_importance_sampling):
        bm = Vec3f(0.0)
        wi = Vec3f(0.0)
        pdf = 0.0
        n = si.normal
        t, b = _get_tangent_space(n)
        if use_importance_sampling:
            bm = si.albedo
            wi = _sample_hemisphere_cosine_weighted(t, b, n)
            pdf = max(0.0, si.normal.dot(wi)) / ti.math.pi
        else:
            wi = _sample_hemisphere_uniform(t, b, n)
            bm = si.albedo * (2 * max(0.0, si.normal.dot(wi)))
            pdf = 1.0 / (2.0 * ti.math.pi)
        if wi.dot(si.geo_normal) < 0:
            wi = _reflect(wi, si.geo_normal)
        return bm, wi, pdf
