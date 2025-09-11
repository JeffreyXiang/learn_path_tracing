import taichi as ti


Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f)
HitRecord = ti.types.struct(point=Vec3f, normal=Vec3f, t=ti.f32)
