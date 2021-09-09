import taichi as ti

Mytype = ti.types.struct(val=ti.i32)

a = Mytype(0)
a.val = 1
