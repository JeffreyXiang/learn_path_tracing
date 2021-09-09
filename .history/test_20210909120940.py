import taichi as ti

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
ti.init(device_memory_GB=8, use_unified_memory=False, arch=ti.gpu)

Mytype = ti.types.struct(val=ti.i32)

a = Mytype(0)
b = Mytype.field(shape=())

print(a)
a.val = [1]
print(a)
b[None] = a
