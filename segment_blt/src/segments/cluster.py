from dataclasses import dataclass, field
import mitsuba as mi
import numpy as np

@dataclass
class Cluster:
    """
    paper does sorted keys voxel grid, so I'm doing the same
    could also just do a spatial hash if you want
    """
    segments: list = field(default_factory=list)
    scene_aabb: mi.BoundingBox3f = None
    c: int = 10 # expected number of points / voxel
    alpha: float = 2/3 # for progressive 

    # need these to know which segment each segment endpoint actually belongs to
    # kinda mimics the c++ pointer method, once we convert to c++ just use pointers lol
    endpoint_metadata: list = field(default_factory=list)  # (2S, 2) — (segment_idx, which_end)
    
    # outputs
    # contiguous indices of surface points
    sorted_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    cluster_ranges: list = field(default_factory=list)  # list of (start, end) into sorted_indices

    _iteration: int = 0
    _voxel_size_0: float = 0.0 # init length
    voxel_size: float = 0.0 # THIS IS SIDE LENGTH!!!
    scene_vol: float = 0.0

    # one scalar per cluster, same indexing as rest
    cluster_kernel_values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))

    def set_segments(self, segments):
        self.segments = segments

    def set_scene_aabb(self, aabb):
        self.scene_aabb = aabb
        self.scene_vol = aabb.volume()

    def _compute_voxel_size(self):
        return (self.c * self.scene_vol * (1 / (len(self.segments) * 2))) ** (1/3)
    
    def _compute_progressive_voxel_size(self):
        """
        set voxel_size_0 via compute voxel size based on c
        then each progressive one should be reduced (built in kernel!!)

        its a Power-law decay: r_n = r_0 * n^(-alpha/2)  (Knaus-Zwicker 2011, Eq. 5)
        basically n is the iteration, that makes it slower decay than exponential
        """
        if self._iteration == 0: 
            return self._voxel_size_0
        return self._voxel_size_0 * (self._iteration ** (-self.alpha / 2))

    def _compute_octants(self, normals: np.ndarray) -> np.ndarray:
        """
        normals: (N, 3)
        Returns (N,) int in [0, 5] — dominant axis * 2 + sign
        """
        abs_n = np.abs(normals)
        axis = np.argmax(abs_n, axis=1)
        sign = (normals[np.arange(len(normals)), axis] >= 0).astype(np.int64)
        return axis * 2 + sign

    def _generate_cluster_keys(self, positions, normals, rng: np.random.Generator):
        """
        generates cluster keys for all endpoints

        the packing method uses 20 bits per axis and 3 per octant which means we 
        have ±524288 voxels per so that should be solid for now imo its just speedup
        """

        iter_seed = int(rng.integers(0, 2**31))

        quant = np.round(positions / (self.voxel_size * 1e-3)).astype(np.int64)  # (N, 3)

        # jitter in voxel units: max displacement r/2 in each axis
        # this randomizes boundary assignment each iteration
        def hash3(qx, qy, qz, seed):
            x = qx * np.int64(2654435761) ^ qy * np.int64(805459861) ^ qz ^ np.int64(seed)
            x = ((x >> np.int64(16)) ^ x) * np.int64(0x45d9f3b37)
            x = ((x >> np.int64(16)) ^ x) * np.int64(0x45d9f3b37)
            return (x >> np.int64(16)) ^ x

        hx = hash3(quant[:, 0], quant[:, 1], quant[:, 2], iter_seed)
        hy = hash3(quant[:, 1], quant[:, 2], quant[:, 0], iter_seed + 1)
        hz = hash3(quant[:, 2], quant[:, 0], quant[:, 1], iter_seed + 2)

        scale = self.voxel_size * 0.5 / float(2**63)
        jitter = np.stack([
            hx.astype(np.float64) * scale,
            hy.astype(np.float64) * scale,
            hz.astype(np.float64) * scale,
        ], axis=1)

        jittered = positions + jitter
        coords   = np.floor(jittered / self.voxel_size).astype(np.int64)
        octants  = self._compute_octants(normals)

        # now we fill  
        OFFSET = np.int64(1 << 19)
        i = (coords[:, 0] + OFFSET).astype(np.uint64)
        j = (coords[:, 1] + OFFSET).astype(np.uint64)
        k = (coords[:, 2] + OFFSET).astype(np.uint64)
        o = octants.astype(np.uint64)

        return (i << np.uint64(43)) | (j << np.uint64(23)) | (k << np.uint64(3)) | o              
        """
        the above is equiv to this but its faster because we hate loops ig
        keys = np.empty(len(positions), dtype=np.uint64)
        for idx in range(len(positions)):
            i = np.uint64(coords[idx, 0] + offset)
            j = np.uint64(coords[idx, 1] + offset)
            k = np.uint64(coords[idx, 2] + offset)
            octant = octants[idx]

            keys[idx] = (i << 43) | (j << 23) | (k << 3) | octant

        return keys"""
    
    def _find_cluster_ranges(self, sorted_keys: np.ndarray) -> list:
        """
        sorted_keys: (N,) uint64, already sorted
        Returns list of (start, end) index pairs.
        """
        boundaries = np.where(np.diff(sorted_keys) != 0)[0] + 1
        starts = np.concatenate([[0], boundaries])
        ends   = np.concatenate([boundaries, [len(sorted_keys)]])
        return list(zip(starts.tolist(), ends.tolist()))
    
    def _flatten_endpoints(self):
        """
        Unpack all segment endpoints into flat arrays.
        THIS ENSURES THAT WE HAVE THE ASSOCIATED SEGMENT FOR A GIVEN 

         flatten ie we want it like this
        flat_idx, seg_idx, and which end
        flat_idx | seg_idx | which_end | position        | normal
        0        | 0       | 0 (x)     | [1.1, 2.3, 0.5] | [0, 1, 0]
        1        | 0       | 1 (y)     | [1.2, 2.4, 0.6] | [0, 1, 0]
        2        | 1       | 0 (x)     | [5.0, 1.0, 3.0] | [1, 0, 0]
        3        | 1       | 1 (y)     | [5.1, 1.1, 3.1] | [1, 0, 0]
        
        """
        num_segments = len(self.segments)

        # extract all at once via list comprehension — one python loop, no inner indexing
        x_positions = [[s.x.p.x, s.x.p.y, s.x.p.z] for s in self.segments]  # (S, 3)
        y_positions = [[s.y.p.x, s.y.p.y, s.y.p.z] for s in self.segments]  # (S, 3)
        x_normals   = [[s.x.n.x, s.x.n.y, s.x.n.z] for s in self.segments]  # (S, 3)
        y_normals   = [[s.y.n.x, s.y.n.y, s.y.n.z] for s in self.segments]  # (S, 3)

        # interleave x and y endpoints: [x0, y0, x1, y1, ...]
        # np.empty then fill by stride is faster than concatenate + reshape
        # 2 (* num_segment) because we have 2 endpoints per segment
        positions = np.empty((num_segments * 2, 3), dtype=np.float64)
        normals   = np.empty((num_segments * 2, 3), dtype=np.float64)
        positions[0::2] = x_positions   # even rows = x endpoints
        positions[1::2] = y_positions   # odd rows  = y endpoints
        normals[0::2]   = x_normals
        normals[1::2]   = y_normals

        # endpoint_meta: flat_idx = seg_idx * 2 + which_end
        # so seg_idx  = flat_idx // 2
        #    which_end = flat_idx % 2
        # we can compute this without a loop at all
        seg_indices = np.repeat(np.arange(num_segments, dtype=np.int32), 2)   # [0,0,1,1,2,2,...]
        which_ends  = np.tile([0, 1], num_segments).astype(np.int32)           # [0,1,0,1,0,1,...]
        self.endpoint_metadata = np.stack([seg_indices, which_ends], axis=1) # (2S, 2)
        return positions, normals
    
    def compute_cluster_stats(self, positions: np.ndarray, normals: np.ndarray):
        """
        Compute per-cluster mean position and mean normal.
        These are consumed by area estimation to approximate K.

        Must be called after cluster(), passing the same positions/normals
        used during key generation (pre-jitter originals, not jittered).

        Sets:
            self.cluster_mean_positions: (C, 3) float64
            self.cluster_mean_normals:   (C, 3) float64  (not re-normalized)
        """
        C = len(self.cluster_ranges)
        self.cluster_mean_positions = np.empty((C, 3), dtype=np.float64)
        self.cluster_mean_normals   = np.empty((C, 3), dtype=np.float64)

        for c_idx, (start, end) in enumerate(self.cluster_ranges):
            flat_indices = self.sorted_indices[start:end]         # endpoints in this cluster
            self.cluster_mean_positions[c_idx] = positions[flat_indices].mean(axis=0)
            raw_normals = normals[flat_indices].mean(axis=0)
            self.cluster_mean_normals[c_idx] = raw_normals / np.linalg.norm(raw_normals)
            # note: mean normal must be renormalized because the mean will not be a unit vector
            # from all normals


    def cluster(self, rng: np.random.Generator, voxel_size = 0):
        # give cluster as variable since each iteration clusters and has a differetn jitter offset
        assert len(self.segments) > 0, "No segments set"
        assert self.c > 0, "Target cluster size c must be positive"

        if (voxel_size <= 0):
            if self._iteration == 0:
                self._voxel_size_0 = self._compute_voxel_size()
            self.voxel_size = self._compute_progressive_voxel_size()
        else:
            self.voxel_size = voxel_size

        positions, normals = self._flatten_endpoints()

        keys = self._generate_cluster_keys(positions, normals, rng)

        # sort
        """
        each endpoint has an associated keys within the 2S keys array
        this sorts so that endpoints with the same key are contiguous
        But the keys just contains the index from the original flattened endpoint
        arrays. Sorted indices doesn't actually contain any key information, just 
        means that any endpoints sharing the same key are next to eachother.

        we need to pass over the sorted_keys array AGAIN to find the ranges
        where each key starts and ends!!!!
        """
        self.sorted_indices = np.argsort(keys, kind='stable').astype(np.int32)
        sorted_keys = keys[self.sorted_indices]

        self.cluster_ranges = self._find_cluster_ranges(sorted_keys)
    
        self.compute_cluster_stats(positions, normals)
        self.build_reverse_map()

    def get_cluster_segments(self, cluster_idx: int) -> list:
        """
        useful for tests, return all (segment, which_end) pairs in a cluster.

        Returns list of (Segment, int) where int is 0 for x endpoint, 1 for y.
        """
        start, end = self.cluster_ranges[cluster_idx]
        result = []
        for pos in range(start, end):
            flat_idx              = self.sorted_indices[pos]
            seg_idx, which_end    = self.endpoint_metadata[flat_idx]
            result.append((self.segments[seg_idx], int(which_end)))
        return result

    def build_reverse_map(self):
        self.endpoint_to_cluster = np.empty(len(self.segments) * 2, dtype=np.int32)
        for c_idx, (start, end) in enumerate(self.cluster_ranges):
            self.endpoint_to_cluster[self.sorted_indices[start:end]] = c_idx

    def get_y_cluster(self, seg_idx: int) -> int:
        """Returns the cluster index containing y-endpoint of segment seg_idx.
        This is just for testing"""
        flat_idx = seg_idx * 2 + 1  # which_end=1 is y
        return self.endpoint_to_cluster[flat_idx]
    
    def compute_cluster_kernel_values(self):
        # plane with normal n slicing through axis aligned cube of side a
        # what is the area of the polygon cut out?

        # According tot he paper: The kernel value is then approximated as
        # the reciprocal of the intersection area between this plane and the voxel
        # Area projected = A(plane) * | n dot e|
        """
        here e is the most aligned axis, dot n to the most aligned axis (ig gives how much its tilted)
        A plane has a min of l * l, max of l * (corner to corner distance)
        we make an approximation that one of the sides of the plane always coincides (ie the projected
        plane is just tilted on one axis) 

        | -> /

        Obviously in this case, you just have the tilted side l / cos tilt angle
        that cos tilt angle is just n * the most aligned axis lol
        """
        abs_n = np.abs(self.cluster_mean_normals)
        tilt = abs_n.max(axis=1)
        self.cluster_kernel_values =  tilt / (self.voxel_size ** 2)