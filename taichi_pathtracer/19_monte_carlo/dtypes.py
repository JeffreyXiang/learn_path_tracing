import taichi as ti


Vec2i = ti.types.vector(2, int)
Vec2f = ti.types.vector(2, float)
Vec3f = ti.types.vector(3, float)
Vec3u8 = ti.types.vector(3, ti.u8)
Mat3f = ti.types.matrix(3, 3, float)
Ray = ti.types.struct(ro=Vec3f, rd=Vec3f)
SurfaceInteraction = ti.types.struct(
    bsdf_id=ti.i32,
    point=Vec3f,
    normal=Vec3f,
    geo_normal=Vec3f,
    albedo=Vec3f,
    metallic=ti.f32,
    roughness=ti.f32,
    ior=ti.f32,
    transparency=ti.f32,    
    emission=Vec3f,
)
AABB = ti.types.struct(low=Vec3f, high=Vec3f)
BVHTraverseStatistics = ti.types.struct(
    nfe_aabb=ti.i32,
    nfe_primitive=ti.i32,
)

class PrimitiveType:
    UNHIT = 0
    SPHERE = 1
    TRIANGLE = 2
    LIGHT = 3
