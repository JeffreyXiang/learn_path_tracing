from tqdm import tqdm
import numpy as np
import taichi as ti
from dtypes import AABB, BVHTraverseStatistics


INF = 114514114514
BVHNode = ti.types.struct(left=ti.i32, right=ti.i32, aabb=AABB, data=ti.i32, size=ti.i32, next=ti.i32)


class BVHSplitMode:
    MIDDLE = 'middle'
    EQUAL = 'equal'
    SAH = 'sah'


@ti.func
def aabb_hit(aabb, ray):
    invdir = 1 / ray.rd
    i = (aabb.low - ray.ro) * invdir
    o = (aabb.high - ray.ro) * invdir
    tmax = ti.max(i, o)
    tmin = ti.min(i, o)
    t1 = ti.min(tmax[0], ti.min(tmax[1], tmax[2]))
    t0 = ti.max(tmin[0], ti.max(tmin[1], tmin[2]))
    return t1 >= t0 and t1 > 0, t0


@ti.data_oriented
class BVH:
    def __init__(self, primitive_type, hit_record_type, size):
        size = max(size, 16)
        self.primitive_type = primitive_type
        self.hit_record_type = hit_record_type
        self.nodes = BVHNode.field(shape=(2 * size - 1,))
        self.primitives = primitive_type.field(shape=(size,))
        self.node_cnt = ti.field(ti.i32, shape=())
        self.primitive_cnt = ti.field(ti.i32, shape=())
        self.depth = 0

    @staticmethod
    def split_node_middle(cur_aabb, aabbs, centers):
        # split along the longest axis
        size = cur_aabb[1] - cur_aabb[0]
        axis = np.argmax(size)

        # middle point along axis
        middle = 0.5 * (centers[:, axis].min() + centers[:, axis].max())
        mask = centers[:, axis] < middle

        left_map = np.nonzero(mask)[0]
        right_map = np.nonzero(~mask)[0]

        if len(left_map) == 0 or len(right_map) == 0:
            return False, None, None, None, None

        left_aabbs = aabbs[left_map]
        right_aabbs = aabbs[right_map]
        left_low  = left_aabbs[:, 0].min(axis=0)
        left_high = left_aabbs[:, 1].max(axis=0)
        right_low  = right_aabbs[:, 0].min(axis=0)
        right_high = right_aabbs[:, 1].max(axis=0)

        left_aabb  = np.stack([left_low, left_high])
        right_aabb = np.stack([right_low, right_high])

        return True, left_aabb, right_aabb, left_map, right_map

    @staticmethod
    def split_node_equal(cur_aabb, aabbs, centers):
        # split along the longest axis
        size = cur_aabb[1] - cur_aabb[0]
        axis = np.argmax(size)

        # sort objects along axis, split into equal halves
        sorted_map = np.argsort(centers[:, axis])
        mid = len(sorted_map) // 2
        left_map, right_map = sorted_map[:mid], sorted_map[mid:]

        if len(left_map) == 0 or len(right_map) == 0:
            return False, None, None, None, None

        left_aabbs = aabbs[left_map]
        right_aabbs = aabbs[right_map]
        left_low  = left_aabbs[:, 0].min(axis=0)
        left_high = left_aabbs[:, 1].max(axis=0)
        right_low  = right_aabbs[:, 0].min(axis=0)
        right_high = right_aabbs[:, 1].max(axis=0)

        left_aabb  = np.stack([left_low, left_high])
        right_aabb = np.stack([right_low, right_high])

        return True, left_aabb, right_aabb, left_map, right_map

    @staticmethod
    def split_node_sah(cur_aabb, aabbs, centers):
        size_total = cur_aabb[1] - cur_aabb[0]
        area_total = size_total[0] * size_total[1] + size_total[1] * size_total[2] + size_total[2] * size_total[0]
        min_cost = len(aabbs)
        min_axis = -1
        min_idx = -1
        min_left_aabb = None
        min_right_aabb = None
        min_left_map = None
        min_right_map = None

        # try split along 3 axis
        for axis in range(3):
            # sort objects along the axis
            sorted_map = np.argsort(centers[:, axis])
            sorted_aabbs = aabbs[sorted_map]

            # compare all candidate splits
            # scan aabb for left subtree
            left_aabb_low = np.minimum.accumulate(sorted_aabbs[:, 0], axis=0)[:-1]
            left_aabb_high = np.maximum.accumulate(sorted_aabbs[:, 1], axis=0)[:-1]

            # reverse scan aabb for right subtree
            right_aabb_low = np.minimum.accumulate(sorted_aabbs[::-1, 0], axis=0)[::-1][1:]
            right_aabb_high = np.maximum.accumulate(sorted_aabbs[::-1, 1], axis=0)[::-1][1:]

            # calculate cost for each split
            size_left  = left_aabb_high - left_aabb_low
            size_right = right_aabb_high - right_aabb_low
            area_left  = size_left[:, 0]  * size_left[:, 1]  + size_left[:, 1]  * size_left[:, 2]  + size_left[:, 2]  * size_left[:, 0]
            area_right = size_right[:, 0] * size_right[:, 1] + size_right[:, 1] * size_right[:, 2] + size_right[:, 2] * size_right[:, 0]
            cnt_left   = np.arange(len(sorted_map) - 1) + 1
            cnt_right  = len(sorted_map) - cnt_left
            cost = 1 + (area_left * cnt_left + area_right * cnt_right) / area_total

            # find best split
            axis_min_cost_idx = np.argmin(cost)
            if cost[axis_min_cost_idx] < min_cost:
                min_cost = cost[axis_min_cost_idx]
                min_axis = axis
                min_idx = axis_min_cost_idx
                min_left_aabb = np.stack([left_aabb_low[axis_min_cost_idx], left_aabb_high[axis_min_cost_idx]])
                min_right_aabb = np.stack([right_aabb_low[axis_min_cost_idx], right_aabb_high[axis_min_cost_idx]])
                min_left_map = sorted_map[:axis_min_cost_idx + 1]
                min_right_map = sorted_map[axis_min_cost_idx + 1:]

        if min_axis >= 0:
            return True, min_left_aabb, min_right_aabb, min_left_map, min_right_map
        else:
            return False, None, None, None, None

    @staticmethod
    def split_node(cur_aabb, aabbs, centers, split_mode):
        if split_mode == BVHSplitMode.MIDDLE:
            return BVH.split_node_middle(cur_aabb, aabbs, centers)
        elif split_mode == BVHSplitMode.EQUAL:
            return BVH.split_node_equal(cur_aabb, aabbs, centers)
        elif split_mode == BVHSplitMode.SAH:
            return BVH.split_node_sah(cur_aabb, aabbs, centers)
        else:
            raise ValueError('Invalid split mode')

    def build_dfs_recursive(self, cur_aabb, indices, aabbs, centers, depth, pbar):
        # If it's a leaf node, add leaf node and primitive to the field
        if len(indices) == 1 or depth >= self.building_cache['max_depth']:
            self.depth = max(self.depth, depth)
            cur_node_ptr = self.building_cache['node_ptr']
            self.nodes[cur_node_ptr] = BVHNode(left=-1, right=-1, aabb=AABB(cur_aabb[0], cur_aabb[1]), data=self.building_cache['primitive_ptr'], size=len(indices))
            self.building_cache['node_ptr'] += 1
            self.nodes[cur_node_ptr].next = self.building_cache['node_ptr']
            for i in indices:
                self.primitives[self.building_cache['primitive_ptr']] = self.building_cache['primitives'][i]
                self.building_cache['primitive_ptr'] += 1
                pbar.update(1)
            return cur_node_ptr

        # Add node to the field
        cur_node_ptr = self.building_cache['node_ptr']
        self.nodes[cur_node_ptr] = BVHNode(left=-1, right=-1, aabb=AABB(cur_aabb[0], cur_aabb[1]), data=-1)
        self.building_cache['node_ptr'] += 1

        # Split objects into left and right sub-trees
        need_split, left_aabb, right_aabb, left_map, right_map, \
            = BVH.split_node(cur_aabb, aabbs, centers, self.building_cache['split_mode'])

        if need_split:
            # Recursively build left and right sub-trees
            left_indices, left_aabbs, left_centers = indices[left_map], aabbs[left_map], centers[left_map]
            right_indices, right_aabbs, right_centers = indices[right_map], aabbs[right_map], centers[right_map]
            left_node_ptr = self.build_dfs_recursive(left_aabb, left_indices, left_aabbs, left_centers, depth + 1, pbar)
            self.nodes[cur_node_ptr].left = left_node_ptr
            right_node_ptr = self.build_dfs_recursive(right_aabb, right_indices, right_aabbs, right_centers, depth + 1, pbar)
            self.nodes[cur_node_ptr].right = right_node_ptr
            self.nodes[cur_node_ptr].next = self.building_cache['node_ptr']
        else:
            # If no need to split, add leaf node and primitive to the field
            self.depth = max(self.depth, depth)
            self.nodes[cur_node_ptr].data = self.building_cache['primitive_ptr']
            self.nodes[cur_node_ptr].size = len(indices)
            self.nodes[cur_node_ptr].next = self.building_cache['node_ptr']
            for i in indices:
                self.primitives[self.building_cache['primitive_ptr']] = self.building_cache['primitives'][i]
                self.building_cache['primitive_ptr'] += 1
                pbar.update(1)

        return cur_node_ptr

    def build(self, primitives, max_depth=None, split_mode=BVHSplitMode.SAH, verbose=True):
        # Initialize global variables
        self.building_cache = {
            'primitives': primitives,
            'max_depth': max_depth or INF,
            'split_mode': split_mode,
            'node_ptr': 0,
            'primitive_ptr': 0,
        }
        self.depth = 0
        if len(primitives) > 0:
            pbar = tqdm(total=len(primitives), desc='Building BVH', disable=not verbose)
            # Prepare objects list
            aabbs = [obj.AABB() for obj in primitives]
            aabbs = np.array([[aabb.low, aabb.high] for aabb in aabbs], dtype=np.float32)
            centers = (aabbs[:, 0] + aabbs[:, 1]) / 2
            indices = np.arange(len(primitives))
            # calculate AABB of all primitives
            root_aabb = np.stack([aabbs[:, 0].min(axis=0), aabbs[:, 1].max(axis=0)])
            # Build tree recursively
            self.build_dfs_recursive(root_aabb, indices, aabbs, centers, 1, pbar)
            pbar.close()
        # Set node count and primitive count
        self.node_cnt[None] = self.building_cache['node_ptr']
        self.primitive_cnt[None] = self.building_cache['primitive_ptr']
        del self.building_cache

    def print_node_dfs_recursive(self, node_ptr, depth):
        indent = '  '*(depth-1)
        aabb = f'[{self.nodes[node_ptr].aabb.low[0]:.3f}, {self.nodes[node_ptr].aabb.low[1]:.3f}, {self.nodes[node_ptr].aabb.low[2]:.3f}, ' + \
               f'{self.nodes[node_ptr].aabb.high[0]:.3f}, {self.nodes[node_ptr].aabb.high[1]:.3f}, {self.nodes[node_ptr].aabb.high[2]:.3f}]'
        if self.nodes[node_ptr].data >= 0:
            print(indent, 'AABB: ', aabb, '  SIZE: ', self.nodes[node_ptr].size)
        else:
            print(indent, 'AABB: ', aabb)
            self.print_node_dfs_recursive(self.nodes[node_ptr].left, depth + 1)
            self.print_node_dfs_recursive(self.nodes[node_ptr].right, depth + 1)

    def print(self):
        print('BVH:')
        print('Depth: ', self.depth)
        print('Nodes: ')
        self.print_node_dfs_recursive(0, 1)

    @ti.func
    def hit(self, ray):
        hit_id = -1
        hit_t = -1.0
        hit_record = self.hit_record_type(0.0)
        # No stack traverse
        ptr = 0
        # Visualization
        vis = BVHTraverseStatistics()
        # Traverse BVH tree
        while ptr < self.node_cnt[None]:
            vis.nfe_aabb += 1
            aabb_in, aabb_t = aabb_hit(self.nodes[ptr].aabb, ray)
            if aabb_in and (hit_t < 0 or aabb_t < hit_t):
                if self.nodes[ptr].data >= 0:
                    vis.nfe_primitive += self.nodes[ptr].size
                    for i in range(self.nodes[ptr].data, self.nodes[ptr].data + self.nodes[ptr].size):
                        record = self.primitives[i].hit(ray)
                        if record.t >= 1e-4 and (hit_t < 0 or record.t < hit_t):
                            hit_id = i
                            hit_t = record.t
                            hit_record = record
                    ptr = self.nodes[ptr].next
                else:
                    ptr = self.nodes[ptr].left
            else:
                ptr = self.nodes[ptr].next
        return hit_id, hit_record, vis
