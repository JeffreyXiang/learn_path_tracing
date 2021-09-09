import taichi as ti

ti.init(arch=ti.cpu)

@ti.func
def array_get(a: ti.template(), i):
    ret = 0
    for j in ti.static(range(len(a))):
        if i == j:
            ret = a[j]
    return ret

@ti.func
def array_set(a: ti.template(), i, val):
    ret = 0
    for j in ti.static(range(len(a))):
        if i == j:
            a[j] = val

@ti.kernel
def func():
    x = ti.Vector.zero(ti.i32, 8)
    print(x)
    array_set(x, 0, 1)
    print(x)
    print(array_get(x, 0))

func()
