import taichi as ti

Mytype = ti.types.struct(val=ti.i32)

a = Mytype(0)
print(a)
a.val = [1]
print(a)
