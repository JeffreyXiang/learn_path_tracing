import taichi as ti
from dtypes import Vec3f


@ti.func
def _get_tangent_space(n):
    # Frisvad’s method
    t = Vec3f(0.0)
    b = Vec3f(0.0)
    if n[2] < -0.9999999:
        t[1] = -1.0
        b[0] = -1.0
    else:
        a = 1.0 / (1.0 + n[2])
        bb = -n[0] * n[1] * a
        t[0] = 1.0 - n[0] * n[0] * a
        t[1] = bb
        t[2] = -n[0]
        b[0] = bb
        b[1] = 1 - n[1] * n[1] * a
        b[2] = -n[1]
    return t, b

@ti.func
def _sample_hemisphere_uniform(t, b, n):
    u1 = ti.random(ti.f32)
    u2 = ti.random(ti.f32)
    z = u1
    phi = 2.0 * ti.math.pi * u2
    r = ti.sqrt(1.0 - z * z)
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    local_dir = Vec3f([x, y, z])
    return t * local_dir.x + b * local_dir.y + n * local_dir.z


@ti.func
def _sample_hemisphere_cosine_weighted(t, b, n):
    u1 = ti.random(ti.f32)
    u2 = ti.random(ti.f32)
    r = ti.sqrt(u1)
    phi = 2.0 * ti.math.pi * u2
    x = r * ti.cos(phi)
    y = r * ti.sin(phi)
    z = ti.sqrt(max(0.0, 1.0 - r * r))
    local_dir = Vec3f([x, y, z])
    return t * local_dir.x + b * local_dir.y + n * local_dir.z


@ti.func
def _schlick_fresnel(cos_theta):
    # F0: Vec3f, cos_theta: scalar
    # Schlick approximation
    one_minus = 1.0 - cos_theta
    factor = one_minus * one_minus
    factor = factor * factor * one_minus
    return factor


@ti.func
def _reflect(dir, normal):
    k = -dir.dot(normal)
    r = dir + 2 * k * normal
    return r


@ti.func
def _D_GGX(ndoth, alpha):
    out = 0.0
    if ndoth > 0.0:
        a2 = alpha * alpha
        ndoth2 = ndoth * ndoth
        denom = ndoth2 * (a2 - 1.0) + 1.0
        denom = ti.math.pi * denom * denom
        out = a2 / max(denom, 1e-16)
    return out
    

@ti.func
def _G1_GGX(ndotv, alpha):
    out = 0.0
    if ndotv > 0.0:
        # tan^2 = (1 - cos^2) / cos^2
        cos2 = ndotv * ndotv
        sin2 = ti.max(0.0, 1.0 - cos2)
        tan2 = sin2 / cos2
        a2 = alpha * alpha
        # Smith's masking function for GGX
        out = 2.0 / (1.0 + ti.sqrt(1.0 + a2 * tan2))
    return out


@ti.func
def _G_smith(cos_i, cos_o, alpha):
    return _G1_GGX(cos_i, alpha) * _G1_GGX(cos_o, alpha)


@ti.func
def _sample_hemisphere_ggx_h(t, b, n, alpha):
    u1 = ti.random(ti.f32)
    u2 = ti.random(ti.f32)
    cos2_theta = (1.0 - u1) / (1.0 + (alpha*alpha - 1.0) * u1)
    cos_theta = ti.sqrt(cos2_theta)
    sin_theta = ti.sqrt(1.0 - cos2_theta)
    phi = 2.0 * ti.math.pi * u2
    local_dir = Vec3f([sin_theta * ti.cos(phi), sin_theta * ti.sin(phi), cos_theta])
    return t * local_dir.x + b * local_dir.y + n * local_dir.z


@ti.func
def _lum(c):
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z


class PrincipledBSDF:
    @staticmethod
    @ti.func
    def f(wo, wi, si: ti.template()):
        f = Vec3f(0.0)
        n = si.normal
        h = (wi + wo).normalized()
        cos_o = n.dot(wo)
        cos_i = n.dot(wi)
        cos_half = h.dot(wi)
        ndoth = n.dot(h)
        f_factor = _schlick_fresnel(cos_half)
        Fd0 = ((si.ior - 1) / (si.ior + 1))**2
        
        # Reflection
        denom = 4.0 * cos_i * cos_o
        
        if denom > 1e-8:
            alpha = si.roughness * si.roughness
            F0 = si.metallic * si.albedo + (1.0 - si.metallic) * Fd0
            ## normal distribution term
            D = _D_GGX(ndoth, alpha)
            ## Fresnel term
            F = F0 + (1.0 - F0) * f_factor
            ## geometry term
            G = _G_smith(cos_i, cos_o, alpha)
            f += (D * F * G) / denom
            
        # Diffuse
        Fd = Fd0 + (1.0 - Fd0) * f_factor
        f += (1.0 - si.metallic) * (1 - Fd) * si.albedo / ti.math.pi
        
        return f
    
    @staticmethod
    @ti.func
    def pdf(wo, wi, si: ti.template(), use_importance_sampling):
        pdf = 0.0
        n = si.normal
        if use_importance_sampling:
            cos_o = n.dot(wo)
            Fd0 = ((si.ior - 1) / (si.ior + 1))**2
            F0 = si.metallic * si.albedo + (1.0 - si.metallic) * Fd0
            
            # mirror reflection specially
            if si.roughness == 0.0:
                # ignore delta distribution
                f_factor = _schlick_fresnel(cos_o)
                F = F0 + (1.0 - F0) * f_factor
                Fd = Fd0 + (1.0 - Fd0) * f_factor
                T = (1 - si.metallic) * (1 - Fd) * si.albedo
                F_lum = _lum(F)
                T_lum = _lum(T)
                p_F = F_lum / (F_lum + T_lum)
                p_T = 1.0 - p_F
                pdf = p_T * wi.dot(n) / ti.math.pi
                
            # ggx + lambertian
            else:
                T0 = (1.0 - si.metallic) * (1.0 - Fd0) * si.albedo
                F_lum = _lum(F0)
                T_lum = _lum(T0)
                p_F = F_lum / (F_lum + T_lum)
                p_T = 1.0 - p_F
                alpha = si.roughness * si.roughness
                h = (wi + wo).normalized()
                cos_i = n.dot(wi)
                cos_half = h.dot(wi)
                ndoth = n.dot(h)                    
                D = _D_GGX(ndoth, alpha)
                pdf_ggx = D * ndoth / (4.0 * cos_half)
                pdf_lambert = cos_i / ti.math.pi
                pdf = p_F * pdf_ggx + p_T * pdf_lambert               
        else:
            pdf = 1.0 / (2.0 * ti.math.pi)
        
        return pdf
    
    @staticmethod
    @ti.func
    def sample(wo, si: ti.template(), use_importance_sampling):
        bm = Vec3f(0.0)
        wi = Vec3f(0.0)
        pdf = 1.0
        n = si.normal
        t, b = _get_tangent_space(n)
        if use_importance_sampling:
            cos_o = n.dot(wo)
            Fd0 = ((si.ior - 1) / (si.ior + 1))**2
            F0 = si.metallic * si.albedo + (1.0 - si.metallic) * Fd0
            
            # mirror reflection specially
            if si.roughness == 0.0:
                f_factor = _schlick_fresnel(cos_o)
                F = F0 + (1.0 - F0) * f_factor
                Fd = Fd0 + (1.0 - Fd0) * f_factor
                T = (1 - si.metallic) * (1 - Fd) * si.albedo
                F_lum = _lum(F)
                T_lum = _lum(T)
                p_F = F_lum / (F_lum + T_lum)
                p_T = 1.0 - p_F
                if ti.random(ti.f32) < p_F:
                    wi = _reflect(-wo, n)
                    pdf = ti.math.inf
                    bm = F / p_F
                else:
                    wi = _sample_hemisphere_cosine_weighted(t, b, n)
                    pdf = p_T * wi.dot(n) / ti.math.pi
                    bm = T / p_T
            
            # ggx + lambertian
            else:
                T0 = (1.0 - si.metallic) * (1.0 - Fd0) * si.albedo
                F_lum = _lum(F0)
                T_lum = _lum(T0)
                p_F = F_lum / (F_lum + T_lum)
                p_T = 1.0 - p_F
                alpha = si.roughness * si.roughness
            
                # Sample wi with mixture of ggx and lambertian
                h = Vec3f(0.0)
                if ti.random(ti.f32) < p_F:
                    h = _sample_hemisphere_ggx_h(t, b, n, alpha)
                    wi = _reflect(-wo, h)
                else:
                    wi = _sample_hemisphere_cosine_weighted(t, b, n)
                    h = (wi + wo).normalized()
                
                # Compute F and pdf
                if n.dot(wi) > 0:
                    cos_i = n.dot(wi)
                    cos_half = h.dot(wi)
                    ndoth = n.dot(h)
                    f_factor = _schlick_fresnel(cos_half)
                    
                    # Reflection                
                    ## normal distribution term
                    D = _D_GGX(ndoth, alpha)
                    ## Fresnel term
                    F = F0 + (1.0 - F0) * f_factor
                    ## geometry term
                    G = _G_smith(cos_i, cos_o, alpha)
                    f_ggx = (D * F * G) / (4.0 * cos_i * cos_o)
                        
                    # Diffuse
                    Fd = Fd0 + (1.0 - Fd0) * f_factor
                    f_lambert = (1.0 - si.metallic) * (1 - Fd) * si.albedo / ti.math.pi
                    
                    f = f_ggx + f_lambert
                    
                    pdf_ggx = D * ndoth / (4.0 * cos_half)
                    pdf_lambert = cos_i / ti.math.pi
                    pdf = p_F * pdf_ggx + p_T * pdf_lambert
                                        
                    bm = f * cos_i / pdf 
        else:
            wi = _sample_hemisphere_uniform(t, b, n)
            f = PrincipledBSDF.f(wo, wi, si)
            bm = f * wi.dot(n) * (2.0 * ti.math.pi)
            pdf = 1.0 / (2.0 * ti.math.pi)
        if wi.dot(si.geo_normal) < 0:
            wi = _reflect(wi, si.geo_normal)
        return bm, wi, pdf