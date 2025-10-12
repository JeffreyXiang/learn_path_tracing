import numpy as np
import taichi as ti
from dtypes import Vec3f, Ray, PrimitiveType
from bsdf import PrincipledBSDF
from alias_table import AliasTable


class PathIntegrator:
    @ti.func
    def run(self, ray: ti.template(), world: ti.template()):
        pass
    
    def prepare(self, world):
        pass


@ti.data_oriented
class RandomWalkPathIntegrator(PathIntegrator):
    def __init__(
        self,
        propagate_limit,
        BSDF_importance_sampling: bool = True,
    ):
        self.propagate_limit = propagate_limit
        self.BSDF_importance_sampling = BSDF_importance_sampling
        
                
    @ti.func
    def run(self, ray: ti.template(), world: ti.template()):
        bounce = 0
        l = Vec3f(0.0)
        beta = Vec3f(1.0)
        
        while True:
            # 1. intersection test with the world
            record = world.hit(ray)
            
            # 2. if not hit, add environment light, terminate
            if record.prim_type == PrimitiveType.UNHIT:
                l += beta * world.env.sample(ray.rd)
                break
                
            # 3. add light emission at hit point
            l += beta * record.si.emission
            
            # 4. if bounce limit reached, terminate
            bounce += 1
            if bounce >= self.propagate_limit:
                break
                
            # 5. sample BSDF
            bm, wi, pdf = PrincipledBSDF.sample(-ray.rd, record.si, use_importance_sampling=self.BSDF_importance_sampling)
            ray.ro = record.si.point
            ray.rd = wi
            beta *= bm
                
            # 6. if beta is zero, terminate
            if beta[0] < 1e-8 and beta[1] < 1e-8 and beta[2] < 1e-8:
                break
                
        return l


@ti.data_oriented
class NextEventEstimationPathIntegrator(PathIntegrator):
    def __init__(
        self,
        propagate_limit,
        BSDF_importance_sampling: bool = True,
        nee_multi_importance_sampling: bool = True,
    ):
        self.propagate_limit = propagate_limit
        self.BSDF_importance_sampling = BSDF_importance_sampling
        self.nee_multi_importance_sampling = nee_multi_importance_sampling
        self.lights_pdf = None
        
    def prepare(self, world):
        lights_power = [world.lights_BVH.primitives[i].power() for i in range(world.lights_BVH.primitive_cnt[None])]
        self.lights_pdf = AliasTable(len(lights_power))
        self.lights_pdf.build(np.array(lights_power, dtype=np.float32))
        
    @ti.func
    def run(self, ray: ti.template(), world: ti.template()):
        bounce = 0
        l = Vec3f(0.0)
        beta = Vec3f(1.0)
        pdf = 0.0
        
        while True:
            # 1. intersection test with the world
            record = world.hit(ray)
            
            # 2. if not hit, add environment light, terminate
            if record.prim_type == PrimitiveType.UNHIT:
                l += beta * world.env.sample(ray.rd)
                break
                
            # 3. add light emission
            # if first hit or not treat as light source
            if bounce == 0 or record.prim_type != PrimitiveType.LIGHT:
                l += beta * record.si.emission
            # else, MIS if multiple importance sampling is enabled
            elif self.nee_multi_importance_sampling:
                mis_weight = 1.0
                if not ti.math.isinf(pdf):  # inf pdf mean mirror reflection, so only use BSDF sampling
                    prob = self.lights_pdf.probs[record.prim_id]
                    light = world.lights_BVH.primitives[record.prim_id]
                    cos_lo = -record.si.normal.dot(ray.rd)
                    light_pdf = prob * (record.t * record.t) / (cos_lo * light.area())
                    light_pdf2 = light_pdf * light_pdf
                    pdf2 = pdf * pdf
                    mis_weight = pdf2 / (pdf2 + light_pdf2)
                l += beta * mis_weight * record.si.emission
            
            # 4. if bounce limit reached, terminate
            bounce += 1
            if bounce >= self.propagate_limit:
                break
        
            # 5. next event estimation: sample a shadow ray
            light_id, prob = self.lights_pdf.sample()
            light = world.lights_BVH.primitives[light_id]
            light_pos, light_normal = light.sample()
            light_dir = light_pos - record.si.point
            distance = light_dir.norm()
            light_dir /= distance
            shadow_ray = Ray(ro=record.si.point, rd=light_dir)
            cos_lo = -light_normal.dot(light_dir)
            cos_li = record.si.normal.dot(light_dir)
            if cos_lo > 0 and cos_li > 0:  # if could be visible
                shadow_record = world.shadow_hit(shadow_ray)
                if shadow_record.prim_type == PrimitiveType.LIGHT and shadow_record.prim_id == light_id and abs(shadow_record.t - distance) < 1e-4:  # if visible
                    light_pdf = prob * (distance * distance) / (cos_lo * light.area())
                    f = PrincipledBSDF.f(-ray.rd, light_dir, record.si)
                    mis_weight = 1.0
                    if self.nee_multi_importance_sampling:
                        pdf = PrincipledBSDF.pdf(-ray.rd, light_dir, record.si, use_importance_sampling=self.BSDF_importance_sampling)
                        light_pdf2 = light_pdf * light_pdf
                        pdf2 = pdf * pdf
                        mis_weight = light_pdf2 / (pdf2 + light_pdf2)
                    l += beta * mis_weight * cos_li * f * light.radiance / light_pdf
                    
            # 6. sample BSDF
            bm, wi, pdf = PrincipledBSDF.sample(-ray.rd, record.si, use_importance_sampling=self.BSDF_importance_sampling)
            ray.ro = record.si.point
            ray.rd = wi
            beta *= bm
                
            # 7. if beta is zero, terminate
            if beta[0] < 1e-8 and beta[1] < 1e-8 and beta[2] < 1e-8:
                break
                
        return l
