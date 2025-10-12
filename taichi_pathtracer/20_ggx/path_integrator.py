import taichi as ti
from dtypes import Vec3f, PrimitiveType
from bsdf import PrincipledBSDF


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
