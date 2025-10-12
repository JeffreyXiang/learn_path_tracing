import numpy as np
import taichi as ti
from dtypes import Vec3f, Ray, PrimitiveType
from bsdf import PrincipledBSDF
from alias_table import AliasTable
from environment import ImageEnvironment


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
        envmap_importance_sampling: bool = True,
    ):
        self.propagate_limit = propagate_limit
        self.BSDF_importance_sampling = BSDF_importance_sampling
        self.nee_multi_importance_sampling = nee_multi_importance_sampling
        self.envmap_importance_sampling = envmap_importance_sampling
        self.has_light = False
        self.lights_pdf = AliasTable(1)
        self.env_support_is = False
        self.envmap_pdf = AliasTable(1)
        self.envmap_width = 1024
        self.envmap_height = 1024
        self.envmap_id = 0
        
    def prepare(self, world):
        # Build alias table for lights
        self.has_light = world.lights_BVH.primitive_cnt[None] > 0
        if self.has_light:
            lights_power = [world.lights_BVH.primitives[i].power() for i in range(world.lights_BVH.primitive_cnt[None])]
            self.lights_pdf = AliasTable(len(lights_power))
            self.lights_pdf.build(np.array(lights_power, dtype=np.float32))

        # Build alias table for envmap
        self.env_support_is = isinstance(world.env, ImageEnvironment)
        if self.envmap_importance_sampling and self.env_support_is:            
            envmap = world.env.map.to_numpy()
            self.envmap_width = envmap.shape[0]
            self.envmap_height = envmap.shape[1]
            envmap_lum = 0.2126*envmap[:,:,0] + 0.7152*envmap[:,:,1] + 0.0722*envmap[:,:,2]
            uv = np.mgrid[0:envmap.shape[0], 0:envmap.shape[1]].astype(np.float32)
            u = (uv[0] + 0.5) / envmap.shape[0]
            v = (uv[1] + 0.5) / envmap.shape[1]
            jacobian = 0.5 * np.pi * np.sin(np.pi * v)
            envmap_lum *= jacobian
            self.envmap_pdf = AliasTable(envmap.shape[0] * envmap.shape[1])
            self.envmap_pdf.build(envmap_lum.reshape(-1))        
        
    @ti.func
    def run(self, ray: ti.template(), world: ti.template()):
        bounce = 0
        l = Vec3f(0.0)
        beta = Vec3f(1.0)
        pdf = 0.0
        envmap_is = self.envmap_importance_sampling and self.env_support_is
        
        while True:
            # 1. intersection test with the world
            record = world.hit(ray)
            
            # 2. add environment light
            if record.prim_type == PrimitiveType.UNHIT:
                # if first hit or envmap not inportance sampling, add environment light
                if bounce == 0 or not envmap_is:
                    l += beta * world.env.sample(ray.rd)
                # else, MIS if multiple importance sampling is enabled
                elif self.nee_multi_importance_sampling:
                    mis_weight = 1.0
                    if not ti.math.isinf(pdf):  # inf pdf mean mirror reflection, so only use BSDF sampling
                        u = (ti.math.atan2(ray.rd[2], ray.rd[0]) / (2.0 * ti.math.pi) + 0.5) * self.envmap_width
                        v = (ti.math.asin(ti.math.clamp(ray.rd[1], -1, 1)) / ti.math.pi + 0.5) * self.envmap_height
                        pix_id = int(u) * self.envmap_height + int(v)
                        prob = self.envmap_pdf.probs[pix_id]
                        cos_theta = ti.math.sqrt(ray.rd[0] * ray.rd[0] + ray.rd[2] * ray.rd[2])
                        light_pdf = prob * (self.envmap_width * self.envmap_height) / (2 * ti.math.pi * ti.math.pi * cos_theta)
                        light_pdf2 = light_pdf * light_pdf
                        pdf2 = pdf * pdf
                        mis_weight = pdf2 / (pdf2 + light_pdf2)
                    l += beta * mis_weight * world.env.sample(ray.rd)
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

            # 5. next event estimation (light)
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

            # 6. next event estimation (envmap)
            if envmap_is:
                pix_id, prob = self.envmap_pdf.sample()
                u = pix_id // self.envmap_height
                v = pix_id % self.envmap_height
                u_norm = (u + ti.random()) / self.envmap_width
                v_norm = (v + ti.random()) / self.envmap_height
                phi = (u_norm - 0.5) * 2 * ti.math.pi
                theta = (v_norm - 0.5) * ti.math.pi
                cos_theta = ti.math.cos(theta)
                light_dir = Vec3f(
                    cos_theta * ti.math.cos(phi),
                    ti.math.sin(theta),
                    cos_theta * ti.math.sin(phi),
                )
                shadow_ray = Ray(ro=record.si.point, rd=light_dir)
                cos_li = record.si.normal.dot(light_dir)
                if cos_li > 0:  # if could be visible
                    shadow_record = world.shadow_hit(shadow_ray)
                    if shadow_record.prim_type == PrimitiveType.UNHIT:
                        light_pdf = prob * (self.envmap_width * self.envmap_height) / (2 * ti.math.pi * ti.math.pi * cos_theta)
                        light_radiance = world.env.sample(light_dir)
                        f = PrincipledBSDF.f(-ray.rd, light_dir, record.si)
                        mis_weight = 1.0
                        if self.nee_multi_importance_sampling:
                            pdf = PrincipledBSDF.pdf(-ray.rd, light_dir, record.si, use_importance_sampling=self.BSDF_importance_sampling)
                            light_pdf2 = light_pdf * light_pdf
                            pdf2 = pdf * pdf
                            mis_weight = light_pdf2 / (pdf2 + light_pdf2)
                        l += beta * mis_weight * cos_li * f * light_radiance / light_pdf

            # 7. sample BSDF
            bm, wi, pdf = PrincipledBSDF.sample(-ray.rd, record.si, use_importance_sampling=self.BSDF_importance_sampling)
            ray.ro = record.si.point
            ray.rd = wi
            beta *= bm
                
            # 8. if beta is zero, terminate
            if beta[0] < 1e-8 and beta[1] < 1e-8 and beta[2] < 1e-8:
                break
                
        return l
