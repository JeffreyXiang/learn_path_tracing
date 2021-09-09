import taichi as ti

ti.init(arch=ti.cpu)

Vec3f = ti.types.vector(3, ti.f32)
Mytype = ti.types.struct(scalar=ti.i32, vector=Vec3f)

a = Vec3f(0)
b = Mytype.field(shape=())

print(a)
# a.scalar += 1
# a.vector = [1.0,1.0,1.0]
# print(a)
b[None] = a
