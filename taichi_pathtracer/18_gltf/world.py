import trimesh
import trimesh.visual
import numpy as np
import taichi as ti
from dtypes import SurfaceInteraction, BVHTraverseStatistics
from primitives import Sphere, SphereHitRecord, Triangle, TriangleHitRecord
from bvh import BVH, BVHSplitMode
from material import Material, MaterialSlots, TextureAtlas, TextureFilterMode, TextureWrapMode


@ti.data_oriented
class World:
    def __init__(self, spheres=[], triangles=[], env=None, texture_atlas_size=(4096, 4096), max_tex_num=128, max_mat_num=128):
        self.spheres = spheres
        self.spheres_BVH = None
        self.triangles = triangles
        self.triangles_BVH = None
        self.env = env
        self.texture_atlas = TextureAtlas(texture_atlas_size, max_tex_num)
        self.material_slots = MaterialSlots(max_mat_num)
    
    def add_sphere(self, sphere):
        self.spheres.append(sphere)

    def add_triangle(self, triangle):
        self.triangles.append(triangle)

    def set_env(self, env):
        self.env = env

    def add_texture(self, tex_array, tex_id=None, filter=TextureFilterMode.LINEAR, wrap=TextureWrapMode.REPEAT):
        return self.texture_atlas.add(tex_array, tex_id, filter, wrap)

    def add_material(self, mat, mat_id=None):
        return self.material_slots.add(mat, mat_id)

    def load_gltf(self, gltf_path):
        scene = trimesh.load(gltf_path)
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

    def build_texture_atlas(self):
        self.texture_atlas.build()

    @ti.func
    def hit(self, ray):
        sphere_hit_id, sphere_hit_record, sphere_vis = self.spheres_BVH.hit(ray)
        triangle_hit_id, triangle_hit_record, triangle_vis = self.triangles_BVH.hit(ray)

        hit_primitive_type = -1
        hit_t = 1e10
        si = SurfaceInteraction()
        vis = BVHTraverseStatistics()
        if sphere_hit_id >= 0 and sphere_hit_record.t < hit_t:
            hit_primitive_type = 0
            hit_t = sphere_hit_record.t
            vis.nfe_aabb += sphere_vis.nfe_aabb
            vis.nfe_primitive += sphere_vis.nfe_primitive
        if triangle_hit_id >= 0 and triangle_hit_record.t < hit_t:
            hit_primitive_type = 1
            hit_t = triangle_hit_record.t
            vis.nfe_aabb += triangle_vis.nfe_aabb
            vis.nfe_primitive += triangle_vis.nfe_primitive

        if hit_primitive_type == 0:
            self.spheres_BVH.primitives[sphere_hit_id].get_surface_interaction(si, ray, sphere_hit_record, self.material_slots, self.texture_atlas)
        elif hit_primitive_type == 1:
            self.triangles_BVH.primitives[triangle_hit_id].get_surface_interaction(si, ray, triangle_hit_record, self.material_slots, self.texture_atlas)

        if ray.rd.dot(si.geo_normal) > 0:
            si.normal = -si.normal
            si.geo_normal = -si.geo_normal
            si.ior = 1 / si.ior

        return sphere_hit_id >= 0 or triangle_hit_id >= 0, si, vis
