import taichi as ti

ti.init(arch=ti.cpu)

x = ti.field(ti.i32)
ti.root.dense(ti.i, 16).bitmasked(ti.i, 1).place(x)
x[0] = 0
x[1] = 1

@ti.kernel
def func():
    for i in x:
        print(i)

func()
x.snode.parent().deactivate_all()
func()