import cv2
import trimesh
import trimesh.visual
import numpy as np
import taichi as ti
from dtypes import SurfaceInteraction, BVHTraverseStatistics, PrimitiveType
from primitives import Sphere, SphereHitRecord, Triangle, TriangleHitRecord
from lights import Light, LightHitRecord
from bvh import BVH, BVHSplitMode
from material import Material, MaterialSlots, TextureAtlas, TextureFilterMode, TextureWrapMode


HitRecord = ti.types.struct(
    prim_type = ti.i32,
    prim_id = ti.i32,
    t = ti.f32,
    si = SurfaceInteraction,
    vis = BVHTraverseStatistics,
)


ShadowHitRecord = ti.types.struct(
    prim_type = ti.i32,
    prim_id = ti.i32,
    t = ti.f32,
)


@ti.data_oriented
class World:
    def __init__(self, spheres=[], triangles=[], lights=[], env=None, texture_atlas_size=(4096, 4096), max_tex_num=128, max_mat_num=128):
        self.spheres = spheres
        self.spheres_BVH = None
        self.triangles = triangles
        self.triangles_BVH = None
        self.lights = lights
        self.lights_BVH = None
        self.env = env
        self.texture_atlas = TextureAtlas(texture_atlas_size, max_tex_num)
        self.material_slots = MaterialSlots(max_mat_num)
    
    def add_sphere(self, sphere):
        self.spheres.append(sphere)

    def add_triangle(self, triangle):
        self.triangles.append(triangle)
        
    def add_light(self, light):
        self.lights.append(light)

    def set_env(self, env):
        self.env = env

    def add_texture(self, tex_array, tex_id=None, filter=TextureFilterMode.LINEAR, wrap=TextureWrapMode.REPEAT):
        return self.texture_atlas.add(tex_array, tex_id, filter, wrap)

    def add_material(self, mat, mat_id=None):
        return self.material_slots.add(mat, mat_id)

    def load_scene(self, scene):
        assert isinstance(scene, trimesh.Scene), "Only trimesh.Scene is supported"
        for mesh in scene.dump():
            if isinstance(mesh, trimesh.Trimesh):
                # Add material
                visual = mesh.visual
                assert isinstance(visual, trimesh.visual.TextureVisuals), "Only trimesh.visual.TextureVisuals is supported"
                assert isinstance(visual.material, trimesh.visual.material.PBRMaterial), "Only trimesh.visual.material.PbrMaterial is supported"

                ## base color
                if visual.material.baseColorFactor is not None:
                    bc_factor = visual.material.baseColorFactor[:3] / 255.0
                else:
                    bc_factor = (1.0, 1.0, 1.0)
                if visual.material.baseColorTexture is not None:
                    bc_tex_array = np.array(visual.material.baseColorTexture)[..., :3]
                    bc_tex_id = self.add_texture(bc_tex_array)
                else:
                    bc_tex_id = -1

                ## metallic roughness
                if visual.material.metallicFactor is not None:
                    m_factor = visual.material.metallicFactor
                else:
                    m_factor = 1.0
                if visual.material.roughnessFactor is not None:
                    r_factor = visual.material.roughnessFactor
                else:
                    r_factor = 1.0
                if visual.material.metallicRoughnessTexture is not None:
                    mr_tex_array = np.array(visual.material.metallicRoughnessTexture)[..., :3]
                    mr_tex_id = self.add_texture(mr_tex_array)
                else:
                    mr_tex_id = -1

                ## normal
                if visual.material.normalTexture is not None:
                    normal_tex_array = np.array(visual.material.normalTexture)[..., :3]
                    normal_tex_id = self.add_texture(normal_tex_array)
                else:
                    normal_tex_id = -1

                mat_id = self.add_material(Material(
                    baseColorFactor=bc_factor,
                    baseColorTexture=bc_tex_id,
                    metallicFactor=m_factor,
                    roughnessFactor=r_factor,
                    metallicRoughnessTexture=mr_tex_id,
                    normalTexture=normal_tex_id,
                    transmissionFactor=0.0,
                    ior=1.5
                ))

                # Add triangle
                triangles = [
                    Triangle(
                        v0=mesh.vertices[mesh.faces[i][0]],
                        v1=mesh.vertices[mesh.faces[i][1]],
                        v2=mesh.vertices[mesh.faces[i][2]],
                        n0=mesh.vertex_normals[mesh.faces[i][0]],
                        n1=mesh.vertex_normals[mesh.faces[i][1]],
                        n2=mesh.vertex_normals[mesh.faces[i][2]],
                        t0=visual.uv[mesh.faces[i][0]],
                        t1=visual.uv[mesh.faces[i][1]],
                        t2=visual.uv[mesh.faces[i][2]],
                        material_id=mat_id
                    ) for i in range(len(mesh.faces))
                ]
                self.triangles.extend(triangles)

    def build_BVH(self, max_depth=None, split_mode=BVHSplitMode.SAH):
        if self.spheres_BVH is None:
            self.spheres_BVH = BVH(Sphere, SphereHitRecord, len(self.spheres))
        self.spheres_BVH.build(self.spheres, max_depth, split_mode)
        if self.triangles_BVH is None:
            self.triangles_BVH = BVH(Triangle, TriangleHitRecord, len(self.triangles))
        self.triangles_BVH.build(self.triangles, max_depth, split_mode)
        if self.lights_BVH is None:
            self.lights_BVH = BVH(Light, LightHitRecord, len(self.lights))
        self.lights_BVH.build(self.lights, max_depth, split_mode)

    def build_texture_atlas(self):
        self.texture_atlas.build()
        
    def save(self, path):
        dump = {
            'spheres_BVH': {
                'nodes': self.spheres_BVH.nodes.to_numpy(),
                'primitives': self.spheres_BVH.primitives.to_numpy(),
                'node_cnt': self.spheres_BVH.node_cnt[None],
                'primitive_cnt': self.spheres_BVH.primitive_cnt[None],
                'depth': self.spheres_BVH.depth,
            },
            'triangles_BVH': {
                'nodes': self.triangles_BVH.nodes.to_numpy(),
                'primitives': self.triangles_BVH.primitives.to_numpy(),
                'node_cnt': self.triangles_BVH.node_cnt[None],
                'primitive_cnt': self.triangles_BVH.primitive_cnt[None],
                'depth': self.triangles_BVH.depth,
            },
            'lights_BVH': {
                'nodes': self.lights_BVH.nodes.to_numpy(),
                'primitives': self.lights_BVH.primitives.to_numpy(),
                'node_cnt': self.lights_BVH.node_cnt[None],
                'primitive_cnt': self.lights_BVH.primitive_cnt[None],
                'depth': self.lights_BVH.depth,
            },
            'texture_atlas': {
                'size': self.texture_atlas.size,
                'atlas': cv2.imencode('.png',self.texture_atlas.atlas.to_numpy())[1].tobytes(),
                'info': self.texture_atlas.info.to_numpy(),
            },
            'material_slots': {
                'slots': self.material_slots.slots.to_numpy(),
            },
        }
        np.savez(path, **dump)

    def load(self, path):
        dump = dict(np.load(path, allow_pickle=True).items())
            
        self.spheres_BVH = BVH(Sphere, SphereHitRecord, dump['spheres_BVH'].item()['primitive_cnt'])
        self.spheres_BVH.nodes.from_numpy(dump['spheres_BVH'].item()['nodes'])
        self.spheres_BVH.primitives.from_numpy(dump['spheres_BVH'].item()['primitives'])
        self.spheres_BVH.node_cnt[None] = dump['spheres_BVH'].item()['node_cnt']
        self.spheres_BVH.primitive_cnt[None] = dump['spheres_BVH'].item()['primitive_cnt']
        self.spheres_BVH.depth = dump['spheres_BVH'].item()['depth']

        self.triangles_BVH = BVH(Triangle, TriangleHitRecord, dump['triangles_BVH'].item()['primitive_cnt'])
        self.triangles_BVH.nodes.from_numpy(dump['triangles_BVH'].item()['nodes'])
        self.triangles_BVH.primitives.from_numpy(dump['triangles_BVH'].item()['primitives'])
        self.triangles_BVH.node_cnt[None] = dump['triangles_BVH'].item()['node_cnt']
        self.triangles_BVH.primitive_cnt[None] = dump['triangles_BVH'].item()['primitive_cnt']
        self.triangles_BVH.depth = dump['triangles_BVH'].item()['depth']

        self.lights_BVH = BVH(Light, LightHitRecord, dump['lights_BVH'].item()['primitive_cnt'])
        self.lights_BVH.nodes.from_numpy(dump['lights_BVH'].item()['nodes'])
        self.lights_BVH.primitives.from_numpy(dump['lights_BVH'].item()['primitives'])
        self.lights_BVH.node_cnt[None] = dump['lights_BVH'].item()['node_cnt']
        self.lights_BVH.primitive_cnt[None] = dump['lights_BVH'].item()['primitive_cnt']
        self.lights_BVH.depth = dump['lights_BVH'].item()['depth']

        self.texture_atlas = TextureAtlas(tuple(dump['texture_atlas'].item()['size']),
                                          max_tex_num=next(iter(dump['texture_atlas'].item()['info'].values())).shape[0])
        self.texture_atlas.atlas.from_numpy(cv2.imdecode(np.frombuffer(dump['texture_atlas'].item()['atlas'], np.uint8), cv2.IMREAD_UNCHANGED))
        self.texture_atlas.info.from_numpy(dump['texture_atlas'].item()['info'])

        self.material_slots = MaterialSlots(max_mat_num=next(iter(dump['material_slots'].item()['slots'].values())).shape[0])
        self.material_slots.slots.from_numpy(dump['material_slots'].item()['slots'])

    @ti.func
    def hit(self, ray):
        record = HitRecord()
        hit_t = 1e10
        
        sphere_hit_id, sphere_hit_record, sphere_vis = self.spheres_BVH.hit(ray, hit_t)
        record.vis.nfe_aabb += sphere_vis.nfe_aabb
        record.vis.nfe_primitive += sphere_vis.nfe_primitive
        
        triangle_hit_id, triangle_hit_record, triangle_vis = self.triangles_BVH.hit(ray, hit_t)
        record.vis.nfe_aabb += triangle_vis.nfe_aabb
        record.vis.nfe_primitive += triangle_vis.nfe_primitive
        
        light_hit_id, light_hit_record, light_vis = self.lights_BVH.hit(ray, hit_t)
        record.vis.nfe_aabb += light_vis.nfe_aabb
        record.vis.nfe_primitive += light_vis.nfe_primitive

        if light_hit_id >= 0:
            record.prim_type = PrimitiveType.LIGHT
            record.prim_id = light_hit_id
            self.lights_BVH.primitives[light_hit_id].get_surface_interaction(record.si, ray, light_hit_record, self.material_slots, self.texture_atlas)
        elif triangle_hit_id >= 0:
            record.prim_type = PrimitiveType.TRIANGLE
            record.prim_id = triangle_hit_id
            self.triangles_BVH.primitives[triangle_hit_id].get_surface_interaction(record.si, ray, triangle_hit_record, self.material_slots, self.texture_atlas)
        elif sphere_hit_id >= 0:
            record.prim_type = PrimitiveType.SPHERE
            record.prim_id = sphere_hit_id
            self.spheres_BVH.primitives[sphere_hit_id].get_surface_interaction(record.si, ray, sphere_hit_record, self.material_slots, self.texture_atlas)
        record.t = hit_t

        hit = sphere_hit_id >= 0 or triangle_hit_id >= 0 or light_hit_id >= 0
        if hit:
            if ray.rd.dot(record.si.geo_normal) > 0:
                record.si.normal = -record.si.normal
                record.si.geo_normal = -record.si.geo_normal
                record.si.ior = 1 / record.si.ior

        return record
    
    @ti.func
    def shadow_hit(self, ray):
        hit_t = 1e10
        sphere_hit_id, sphere_hit_record, sphere_vis = self.spheres_BVH.hit(ray, hit_t)
        triangle_hit_id, triangle_hit_record, triangle_vis = self.triangles_BVH.hit(ray, hit_t)
        light_hit_id, light_hit_record, light_vis = self.lights_BVH.hit(ray, hit_t)
        return light_hit_id, light_hit_record

    @ti.func
    def shadow_hit(self, ray):
        record = ShadowHitRecord()
        hit_t = 1e10
        sphere_hit_id, sphere_hit_record, sphere_vis = self.spheres_BVH.hit(ray, hit_t)
        triangle_hit_id, triangle_hit_record, triangle_vis = self.triangles_BVH.hit(ray, hit_t)
        light_hit_id, light_hit_record, light_vis = self.lights_BVH.hit(ray, hit_t)
        if light_hit_id >= 0:
            record.prim_type = PrimitiveType.LIGHT
            record.prim_id = light_hit_id
            record.t = light_hit_record.t
        elif triangle_hit_id >= 0:
            record.prim_type = PrimitiveType.TRIANGLE
            record.prim_id = triangle_hit_id
            record.t = triangle_hit_record.t
        elif sphere_hit_id >= 0:
            record.prim_type = PrimitiveType.SPHERE
            record.prim_id = sphere_hit_id
            record.t = sphere_hit_record.t
        return record
