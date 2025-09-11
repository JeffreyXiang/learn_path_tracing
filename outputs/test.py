import os
import taichi as ti

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ti.init(device_memory_GB=8, use_unified_memory=False, arch=ti.gpu)

Vec3f = ti.types.vector(3, ti.f32)

@ti.kernel
def test():
    a = Vec3f([0, 0, 1])
    a.cross(a).normalized()
    assert a.norm() > 1 - 1e-3

test()