import taichi as ti

ti.init(arch=ti.cpu)

Vec3i = ti.types.vector(3, ti.i32)

a = Vec3i(0)
b = Vec3i.field(shape=())

print(a)
a = a + 0.1
print(a)
b[None] = a
