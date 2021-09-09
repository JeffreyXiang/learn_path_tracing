import taichi as ti

ti.init(arch=ti.cpu)

Mytype = ti.types.struct(val=ti.i32)

a = Mytype(0)
b = Mytype.field(shape=())

print(a)
a.val = [1]
print(a)
b[None] = a
