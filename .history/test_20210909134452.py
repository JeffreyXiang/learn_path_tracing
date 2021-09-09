import taichi as ti

ti.init(arch=ti.cpu)

Vec3f = ti.types.vector(3, ti.f32)
Vec3i = ti.types.vector(3, ti.i32)
Mytype = ti.types.struct(scalar=ti.i32, vector=Vec3f)

a = Mytype(0.0)
b = Mytype.field(shape=())

print(a)
a.scalar = 1.0
print(a)
b[None] = a
