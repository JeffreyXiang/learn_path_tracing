from tqdm import tqdm
import numpy as np
import taichi as ti
from dtypes import AABB


INF = 114514114514
LightBVHNode = ti.types.struct(parent=ti.i32, left=ti.i32, right=ti.i32, aabb=AABB, data=ti.i32, power=ti.f32)


@ti.func
def AABB_query_weight(aabb, query_point, query_normal):
    radius_v = (aabb.high - aabb.low) * 0.5
    radius2 = radius_v.dot(radius_v)
    center = (aabb.high + aabb.low) * 0.5
    is_inside = all(query_point > aabb.low) and all(query_point < aabb.high)
    cos_theta = 1.0
    if not is_inside:
        flag = query_normal > 0
        aabb_point = ti.select(flag, aabb.high, aabb.low)
        cos_theta = (aabb_point - query_point).normalized().dot(query_normal)
    dist_v = query_point - center
    dist2 = max(radius2, dist_v.dot(dist_v))
    return max(0, cos_theta) / dist2


@ti.data_oriented
class LightBVH:
    def __init__(self, size):
        size = max(size, 16)
        self.nodes = LightBVHNode.field(shape=(2 * size - 1,))
        self.id2node = ti.field(ti.i32, shape=(size))
        self.node_cnt = ti.field(ti.i32, shape=())
        self.depth = 0

    @staticmethod
    def split_node(aabbs, centers, powers):
        min_cost = INF
        min_left_aabb = None
        min_right_aabb = None
        min_left_power = 0.0
        min_right_power = 0.0
        min_left_map = None
        min_right_map = None

        # try split along 3 axis
        for axis in range(3):
            # sort objects along the axis
            sorted_map = np.argsort(centers[:, axis])
            sorted_aabbs = aabbs[sorted_map]
            sorted_powers = powers[sorted_map]

            # compare all candidate splits
            # scan aabb for left subtree
            left_aabb_low = np.minimum.accumulate(sorted_aabbs[:, 0], axis=0)[:-1]
            left_aabb_high = np.maximum.accumulate(sorted_aabbs[:, 1], axis=0)[:-1]
            left_powers = np.cumsum(sorted_powers)[:-1]

            # reverse scan aabb for right subtree
            right_aabb_low = np.minimum.accumulate(sorted_aabbs[::-1, 0], axis=0)[::-1][1:]
            right_aabb_high = np.maximum.accumulate(sorted_aabbs[::-1, 1], axis=0)[::-1][1:]
            right_powers = np.cumsum(sorted_powers[::-1])[::-1][1:]

            # calculate cost for each split
            size_left  = left_aabb_high - left_aabb_low
            size_right = right_aabb_high - right_aabb_low
            area_left  = size_left[:, 0]  * size_left[:, 1]  + size_left[:, 1]  * size_left[:, 2]  + size_left[:, 2]  * size_left[:, 0]
            area_right = size_right[:, 0] * size_right[:, 1] + size_right[:, 1] * size_right[:, 2] + size_right[:, 2] * size_right[:, 0]
            cost = (area_left * left_powers + area_right * right_powers)

            # find best split
            axis_min_cost_idx = np.argmin(cost)
            if cost[axis_min_cost_idx] < min_cost:
                min_cost = cost[axis_min_cost_idx]
                min_left_aabb = np.stack([left_aabb_low[axis_min_cost_idx], left_aabb_high[axis_min_cost_idx]])
                min_right_aabb = np.stack([right_aabb_low[axis_min_cost_idx], right_aabb_high[axis_min_cost_idx]])
                min_left_power = left_powers[axis_min_cost_idx]
                min_right_power = right_powers[axis_min_cost_idx]
                min_left_map = sorted_map[:axis_min_cost_idx + 1]
                min_right_map = sorted_map[axis_min_cost_idx + 1:]

        return min_left_aabb, min_right_aabb, min_left_power, min_right_power, min_left_map, min_right_map

    def build_dfs_recursive(self, cur_aabb, cur_power, indices, aabbs, centers, powers, depth, pbar):
        # If it's a leaf node, add leaf node and primitive to the field
        if len(indices) == 1:
            self.depth = max(self.depth, depth)
            cur_node_ptr = self.building_cache['node_ptr']
            self.nodes[cur_node_ptr] = LightBVHNode(left=-1, right=-1, aabb=AABB(cur_aabb[0], cur_aabb[1]), data=indices[0], power=cur_power)
            self.building_cache['node_ptr'] += 1
            self.id2node[indices[0]] = cur_node_ptr
            pbar.update(1)
            return cur_node_ptr

        # Add node to the field
        cur_node_ptr = self.building_cache['node_ptr']
        self.nodes[cur_node_ptr] = LightBVHNode(left=-1, right=-1, aabb=AABB(cur_aabb[0], cur_aabb[1]), data=-1, power=cur_power)
        self.building_cache['node_ptr'] += 1

        # Split objects into left and right sub-trees
        left_aabb, right_aabb, left_power, right_power, left_map, right_map \
            = LightBVH.split_node(aabbs, centers, powers)

        # Recursively build left and right sub-trees
        left_indices, left_aabbs, left_centers, left_powers = indices[left_map], aabbs[left_map], centers[left_map], powers[left_map]
        right_indices, right_aabbs, right_centers, right_powers = indices[right_map], aabbs[right_map], centers[right_map], powers[right_map]
        left_node_ptr = self.build_dfs_recursive(left_aabb, left_power, left_indices, left_aabbs, left_centers, left_powers, depth + 1, pbar)
        self.nodes[cur_node_ptr].left = left_node_ptr
        self.nodes[left_node_ptr].parent = cur_node_ptr
        right_node_ptr = self.build_dfs_recursive(right_aabb, right_power, right_indices, right_aabbs, right_centers, right_powers, depth + 1, pbar)
        self.nodes[cur_node_ptr].right = right_node_ptr
        self.nodes[right_node_ptr].parent = cur_node_ptr

        return cur_node_ptr

    def build(self, lights, verbose=True):
        # Initialize global variables
        self.building_cache = {
            'node_ptr': 0,
        }
        self.depth = 0
        if lights.shape[0] > 0:
            pbar = tqdm(total=lights.shape[0], desc='Building BVH', disable=not verbose)
            # Prepare objects list
            aabbs = [lights[i].AABB() for i in range(lights.shape[0])]
            aabbs = np.array([[aabb.low, aabb.high] for aabb in aabbs], dtype=np.float32)
            centers = (aabbs[:, 0] + aabbs[:, 1]) / 2
            powers = np.array([lights[i].power() for i in range(lights.shape[0])], dtype=np.float32)
            indices = np.arange(lights.shape[0])
            # calculate AABB and power of all primitives
            root_aabb = np.stack([aabbs[:, 0].min(axis=0), aabbs[:, 1].max(axis=0)])
            root_power = np.sum(powers)
            # Build tree recursively
            self.build_dfs_recursive(root_aabb, root_power, indices, aabbs, centers, powers, 1, pbar)
            pbar.close()
        # Set node count and primitive count
        self.node_cnt[None] = self.building_cache['node_ptr']
        del self.building_cache

    def print_node_dfs_recursive(self, node_ptr, depth):
        indent = '  '*(depth-1)
        aabb = f'[{self.nodes[node_ptr].aabb.low[0]:.3f}, {self.nodes[node_ptr].aabb.low[1]:.3f}, {self.nodes[node_ptr].aabb.low[2]:.3f}, ' + \
               f'{self.nodes[node_ptr].aabb.high[0]:.3f}, {self.nodes[node_ptr].aabb.high[1]:.3f}, {self.nodes[node_ptr].aabb.high[2]:.3f}]'
        
        print(indent, 'AABB: ', aabb)
        if self.nodes[node_ptr].data < 0:
            self.print_node_dfs_recursive(self.nodes[node_ptr].left, depth + 1)
            self.print_node_dfs_recursive(self.nodes[node_ptr].right, depth + 1)

    def print(self):
        print('LightBVH:')
        print('Depth: ', self.depth)
        print('Nodes: ')
        self.print_node_dfs_recursive(0, 1)

    @ti.func
    def sample(self, query_point, query_normal):
        prob = 1.0
        ptr = 0
        # Traverse BVH tree
        while self.nodes[ptr].data == -1:
            cur_node = self.nodes[ptr]
            left_node = self.nodes[cur_node.left]
            right_node = self.nodes[cur_node.right]
            w_l = AABB_query_weight(left_node.aabb, query_point, query_normal) * left_node.power + 1e-8
            w_r = AABB_query_weight(right_node.aabb, query_point, query_normal) * right_node.power + 1e-8
            p_l = w_l / (w_l + w_r)
            p_r = 1 - p_l
            if ti.random() < p_l:
                ptr = cur_node.left
                prob *= p_l
            else:
                ptr = cur_node.right
                prob *= p_r
        return self.nodes[ptr].data, prob
    
    @ti.func
    def query(self, id, query_point, query_normal):
        prob = 1.0
        ptr = self.id2node[id]
        # Traverse BVH tree
        while ptr != 0:
            cur_node = self.nodes[ptr]
            parent_node = self.nodes[cur_node.parent]
            sibling_ptr = 0
            if parent_node.left != ptr:
                sibling_ptr = parent_node.left
            else:
                sibling_ptr = parent_node.right
            sibling_node = self.nodes[sibling_ptr]
            w_c = AABB_query_weight(cur_node.aabb, query_point, query_normal) * cur_node.power + 1e-8
            w_s = AABB_query_weight(sibling_node.aabb, query_point, query_normal) * sibling_node.power + 1e-8
            prob *= w_c / (w_c + w_s)
            ptr = cur_node.parent
        return prob
