import taichi as ti


Vec2f = ti.types.vector(2, float)
Vec3f = ti.types.vector(3, float)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f, l=Vec3f, end=ti.int8)
SurfaceInteraction = ti.types.struct(
    point=Vec3f,
    normal=Vec3f,
    albedo=Vec3f,
    metallic=ti.i32,
    roughness=ti.f32,
    ior=ti.f32,
    transparency=ti.i32
)
Material = ti.types.struct(albedo=Vec3f, roughness=ti.f32, metallic=ti.i32, ior=ti.f32, transparency=ti.i32)
AABB = ti.types.struct(low=Vec3f, high=Vec3f)
BVHTraverseStatistics = ti.types.struct(
    nfe_aabb=ti.i32,
    nfe_primitive=ti.i32,
)
