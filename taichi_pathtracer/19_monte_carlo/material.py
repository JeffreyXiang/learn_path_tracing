import numpy as np
import taichi as ti
from dtypes import Vec3f, Vec3u8, Vec2i


TextureInfo = ti.types.struct(
    filter=ti.u8,
    wrap=ti.u8,
    x=ti.i32,
    y=ti.i32,
    w=ti.i32,
    h=ti.i32,
)

Material = ti.types.struct(
    baseColorFactor=Vec3f,
    baseColorTexture=ti.i32,
    metallicFactor=ti.f32,
    roughnessFactor=ti.f32,
    metallicRoughnessTexture=ti.i32,
    normalTexture=ti.i32,
    transmissionFactor=ti.f32,
    ior=ti.f32,
)


class TextureFilterMode:
    NEAREST = 0
    LINEAR = 1


class TextureWrapMode:
    CLAMP_TO_EDGE = 0
    REPEAT = 1
    MIRRORED_REPEAT = 2


@ti.data_oriented
class TextureAtlas:
    def __init__(self, size, max_tex_num=128):
        self.size = size                                        # (width, height) of the atlas
        self.atlas = Vec3u8.field(shape=size)                   # texture atlas
        self.info = TextureInfo.field(shape=(max_tex_num,))     # texture metadata
        self.textures = []                                      # list of texture metadata
        self.free_rects = []                                    # list of free rectangles for packing
    
    def add(self, tex_array, tex_id=None, filter=TextureFilterMode.LINEAR, wrap=TextureWrapMode.REPEAT):
        tex_size = (tex_array.shape[1], tex_array.shape[0])  # (w, h)
        if tex_id is None:
            existing_ids = [t['id'] for t in self.textures]
            tex_id = 0
            while tex_id in existing_ids:
                tex_id += 1
        self.textures.append({
            'array': tex_array,
            'size': tex_size,
            'id': tex_id,
            'filter': filter,
            'wrap': wrap,
        })
        return tex_id

    def clear(self):
        self.textures = []

    def _allocate_rect(self, tex_size):
        """Find a free rectangle and allocate space for the texture"""
        w, h = tex_size
        for i, (l, b, r, t) in enumerate(self.free_rects):
            if (r - l) >= w and (t - b) >= h:
                # Occupy bottom-left region
                self.free_rects[i] = [l, b + h, r, t]            # top part remains free
                self.free_rects.insert(i, [l + w, b, r, b + h])  # right part remains free
                return (l, b, l + w, b + h)
        return None

    def build(self):
        """Pack all textures into the atlas using simple guillotine algorithm"""
        self.free_rects = [[0, 0, self.size[0], self.size[1]]]
        # Sort by height first, then width (descending)
        self.textures.sort(key=lambda x: x['size'][0], reverse=True)
        self.textures.sort(key=lambda x: x['size'][1], reverse=True)

        for tex in self.textures:
            rect = self._allocate_rect(tex['size'])
            if rect is None:
                raise MemoryError('Texture atlas overflow.')
            tex['rect'] = rect  # store allocated area
            self.info[tex['id']].x = rect[0]
            self.info[tex['id']].y = rect[1]
            self.info[tex['id']].w = rect[2] - rect[0]
            self.info[tex['id']].h = rect[3] - rect[1]
            self.info[tex['id']].filter = tex['filter']
            self.info[tex['id']].wrap = tex['wrap']
            
        # fill in the atlas with texture data
        atlas_array = self.atlas.to_numpy()
        for tex in self.textures:
            l, b, r, t = tex['rect']
            atlas_array[l:r, b:t] = np.flip(tex['array'].transpose(1, 0, 2), axis=1)
        self.atlas.from_numpy(atlas_array)

    @ti.func
    def wrap_texcoord(self, w, u, wrap):
        o = 0
        if wrap == TextureWrapMode.CLAMP_TO_EDGE:
            o = ti.math.clamp(u, 0, w - 1)
        elif wrap == TextureWrapMode.REPEAT:
            o = (u % w + w) % w
        elif wrap == TextureWrapMode.MIRRORED_REPEAT:
            period = (u % (2 * w) + 2 * w) % (2 * w)
            if period < w:
                o = period
            else:
                o = (2 * w - 1) - period
        return o
        
    @ti.func
    def sample(self, tex_id, uv):
        info = self.info[tex_id]
        u, v = uv[0] * info.w, uv[1] * info.h
        color = Vec3f(0, 0, 0)
        if info.filter == TextureFilterMode.NEAREST:
            s, t = int(ti.math.floor(u)), int(ti.math.floor(v))
            s = self.wrap_texcoord(info.w, s, info.wrap) + info.x
            t = self.wrap_texcoord(info.h, t, info.wrap) + info.y
            color = self.atlas[s, t] / 255.0
        else:
            l = ti.math.floor(u - 0.5) + 0.5
            r = l + 1.0
            b = ti.math.floor(v - 0.5) + 0.5
            t = b + 1.0
            w1 = (r - u) * (t - v)
            w2 = (u - l) * (t - v)
            w3 = (r - u) * (v - b)
            w4 = (u - l) * (v - b)
            sl = self.wrap_texcoord(info.w, int(ti.math.floor(l)), info.wrap) + info.x
            sr = self.wrap_texcoord(info.w, int(ti.math.floor(r)), info.wrap) + info.x
            tb = self.wrap_texcoord(info.h, int(ti.math.floor(b)), info.wrap) + info.y
            tt = self.wrap_texcoord(info.h, int(ti.math.floor(t)), info.wrap) + info.y
            c1 = self.atlas[sl, tb]
            c2 = self.atlas[sr, tb]
            c3 = self.atlas[sl, tt]
            c4 = self.atlas[sr, tt]
            color = ti.math.clamp((w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4) / 255.0, 0, 1)
        return color


@ti.data_oriented
class MaterialSlots:
    def __init__(self, max_mat_num=128):
        self.slots = Material.field(shape=(max_mat_num,))
        self.existing_ids = []

    def add(self, material, mat_id=None):
        if mat_id is None:
            mat_id = 0
            while mat_id in self.existing_ids:
                mat_id += 1
        self.slots[mat_id] = material
        self.existing_ids.append(mat_id)
        return mat_id

    def clear(self):
        self.existing_ids = []
    
    @ti.func
    def sample(self, texture_atlas, mat_id, uv, T, B, N):
        material = self.slots[mat_id]

        baseColor = material.baseColorFactor
        if material.baseColorTexture != -1:
            # Base color texture is in sRGB space, so we need to convert it to linear space first
            baseColor *= texture_atlas.sample(material.baseColorTexture, uv) ** 2.2

        metallic = material.metallicFactor
        roughness = material.roughnessFactor
        if material.metallicRoughnessTexture != -1:
            metallicRoughness = texture_atlas.sample(material.metallicRoughnessTexture, uv)
            metallic *= metallicRoughness[2]        # B channel
            roughness *= metallicRoughness[1]       # G channel

        normal = Vec3f(0, 0, 1)
        if material.normalTexture != -1:
            normal = texture_atlas.sample(material.normalTexture, uv)
            normal = normal * 2.0 - 1.0
        # tangent space to world space
        normal = (normal[0] * T + normal[1] * B + normal[2] * N).normalized()

        return baseColor, metallic, roughness, normal, material.ior, material.transmissionFactor
