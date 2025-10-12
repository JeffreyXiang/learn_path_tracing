import taichi as ti
import numpy as np


@ti.data_oriented
class AliasTable:
    def __init__(self, size):
        self.size = size
        self.probs = ti.field(dtype=ti.f32, shape=size)
        self.ratio = ti.field(dtype=ti.f32, shape=size)
        self.alias = ti.field(dtype=ti.i32, shape=size)

    def build(self, probs):
        probs = probs / probs.sum()
        self.probs.from_numpy(probs)
        scaled_probs = probs * self.size
        
        ratio = np.zeros(self.size, dtype=np.float32)
        alias = np.zeros(self.size, dtype=np.int32)
        
        small = []
        large = []
        for i in range(self.size):
            if scaled_probs[i] < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()
            
            ratio[small_idx] = scaled_probs[small_idx]
            alias[small_idx] = large_idx
            
            scaled_probs[large_idx] = (scaled_probs[large_idx] + scaled_probs[small_idx]) - 1.0
            if scaled_probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
        
        for leftover in small + large:
            ratio[leftover] = 1.0
            alias[leftover] = leftover
        
        self.ratio.from_numpy(ratio)
        self.alias.from_numpy(alias)

    @ti.func
    def sample(self):
        s = 0
        u = ti.random()
        idx = ti.cast(u * self.size, ti.i32)
        r = ti.random()
        if r < self.ratio[idx]:
            s = idx
        else:
            s = self.alias[idx]
        return s, self.probs[s]
