import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

# =============================================================================
# GLOBAL TUNABLES
# =============================================================================

# ---- Output ----
OUTPUT_DIR = "sim_outputs"   # folder where all CSVs will be written

# ---- Time ----
TIME_STEP_HR = 1.0           # x-axis will be time (hrs) = step * TIME_STEP_HR

# ---- Grid / domain (µm) ----
EXTENT_UM = 100.0
CONIDIA_SPAWN_BOX_EXTENT_UM = 60.0
N_GRID = 70
RNG_SEED = 34
N_SHELLS = 70  # for hyphal density analysis

# Grid visualization (drawing every grid line at N=70 is too heavy)
GRID_PLOT_STRIDE = 5  # draw grid lines every Nth voxel edge

# ---- Simulation ----
N_STEPS = 10

# ---- Nutrients (g/L) ----
NUTRIENT_SPECS = {
    "Nitrogen":   {"mean": 2.4,       "pm": 0.3},
    "Phosphorus": {"mean": 0.15,      "pm": 0.2},
    "Glucose":    {"mean": 140.0,     "pm": 15.0},
    "Oxygen":     {"mean": 100.0,     "pm": 5.0},
    "Metal ion":  {"mean": 0.000116,  "pm": 0.000003},
}

SIGMA_SCALE = {
    "Nitrogen":   0.33,
    "Phosphorus": 0.33,
    "Glucose":    0.33,
    "Oxygen":     0.33,
    "Metal ion":  0.33,
}
CLAMP_TO_PM_RANGE = True

# ---- Conidia (spores) ----
N_CONIDIA = 4
CONIDIUM_DIAMETER_UM = 6.2
MIN_CONIDIA_SEPARATION_UM = 20.0
MAX_PLACEMENT_TRIES = 5000

MISS_HYPHA_BASE_PROB = 0.01
MISS_HYPHA_PER_CONIDIUM = 0.002
MISS_HYPHA_MAX_PROB = 0.10

# ---- Hypha geometry & growth ----
HYPHA_RADIUS_UM = 1.5     # sensing/inflation radius
GROWTH_STEP_UM = 5.2      # baseline step; actual step scales with N/P
TRAJECTORY_BIAS = 0.35    # bias to keep direction forward

# Growth step scaling clamp (prevents crazy jumps)
MIN_GROWTH_MULT = 0.10
MAX_GROWTH_MULT = 2.00

SURFACE_SHELL_THICKNESS_MULT = 0.75  # shell thickness ~ dx * this

# ---- Growth thresholds + secretion ----
N_GROWTH_THRESHOLD = 0.01
P_GROWTH_THRESHOLD = 0.01
CITRIC_ACID_RATE_G_PER_L_PER_STEP = 0.1  # (kept for backward compatibility; no longer used)

# ---- Action-local nutrient depletion rules ----
DEPLETION_LINK_FACTOR = 0.8
DEPLETION_TIP_FACTOR = 0.1

# ---- Apical branching ----
APICAL_BRANCH_MIN_AGE_STEPS = 4
APICAL_BRANCH_PROB = 0.8
APICAL_BRANCH_ANGLE_DEG_MIN = 50.0
APICAL_BRANCH_ANGLE_DEG_MAX = 70.0

APICAL_N_THRESHOLD_COEFF = 0.85 
APICAL_P_THRESHOLD_COEFF = 0.85 
APICAL_N_THRESHOLD = APICAL_N_THRESHOLD_COEFF * NUTRIENT_SPECS["Nitrogen"]["mean"]
APICAL_P_THRESHOLD = APICAL_P_THRESHOLD_COEFF * NUTRIENT_SPECS["Phosphorus"]["mean"]

# ---- Lateral branching ----
LATERAL_MIN_DISTANCE_FROM_TIP = 10
LATERAL_N_THRESHOLD = NUTRIENT_SPECS["Nitrogen"]["mean"]
LATERAL_P_THRESHOLD = NUTRIENT_SPECS["Phosphorus"]["mean"]
LATERAL_BRANCH_PROB = 0.2

# ---- Diffusion ----
DIFFUSION_RATE = {
    "Nitrogen": 0.12,
    "Phosphorus": 0.12,
    "Glucose": 0.08,
    "Oxygen": 0.15,
    "Metal ion": 0.10,
}
DIFFUSION_NOISE_FRACTION = 0.02

# ---- Plot colors for conidia skeletons ----
SKELETON_COLORS = [
    (1.0, 0.2, 0.2),
    (0.2, 1.0, 0.2),
    (0.2, 0.3, 1.0),
    (1.0, 0.6, 0.1),
    (0.6, 0.0, 0.8),
    (0.2, 0.8, 0.8),
]

# =============================================================================
# NEW: Citric acid yield model
# =============================================================================
# User requested: (grams glucose consumed) * 749 = mg citric acid produced.
CITRIC_MG_PER_G_GLUCOSE = 749.0


# =============================================================================
# CSV HELPERS
# =============================================================================

def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_csv(path, fieldnames, rows):
    _ensure_outdir()
    full = os.path.join(OUTPUT_DIR, path)
    with open(full, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[CSV] wrote: {full}")


# =============================================================================
# GRID
# =============================================================================

class VoxelGrid3D:
    def __init__(self, extent_um=50.0, n=24, origin_um=(0.0, 0.0, 0.0)):
        self.extent_um = float(extent_um)
        self.n = int(n)
        self.origin_um = np.array(origin_um, dtype=float)

        half = self.extent_um / 2.0
        self.min_um = self.origin_um - half
        self.max_um = self.origin_um + half

        self.dx_um = self.extent_um / self.n

        self.x_edges = np.linspace(self.min_um[0], self.max_um[0], self.n + 1)
        self.y_edges = np.linspace(self.min_um[1], self.max_um[1], self.n + 1)
        self.z_edges = np.linspace(self.min_um[2], self.max_um[2], self.n + 1)

        self.x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        self.y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        self.z_centers = (self.z_edges[:-1] + self.z_edges[1:]) / 2

        self.nutrients = {}
        self.solid_mask = np.zeros((self.n, self.n, self.n), dtype=bool)

        # hypha occupancy (node voxels only, coarse)
        self.hypha_mask = np.zeros((self.n, self.n, self.n), dtype=bool)
        self.hypha_owner = np.full((self.n, self.n, self.n), -1, dtype=np.int32)

        # metabolites
        self.metabolites = {
            # NEW: track citric production in mg (per conidium), computed from glucose consumption
            "citric_acid_total_mg": None,
            # Spatial citric acid production tracking (mg per voxel)
            "citric_acid_spatial_mg": np.zeros((self.n, self.n, self.n), dtype=np.float64),
        }

    def rebuild_center_mesh(self):
        Xc, Yc, Zc = np.meshgrid(self.x_centers, self.y_centers, self.z_centers, indexing="ij")
        return Xc, Yc, Zc

    def world_to_index(self, x_um, y_um, z_um):
        x_um, y_um, z_um = float(x_um), float(y_um), float(z_um)
        if (x_um < self.min_um[0] or x_um >= self.max_um[0] or
            y_um < self.min_um[1] or y_um >= self.max_um[1] or
            z_um < self.min_um[2] or z_um >= self.max_um[2]):
            return None
        i = int((x_um - self.min_um[0]) / self.dx_um)
        j = int((y_um - self.min_um[1]) / self.dx_um)
        k = int((z_um - self.min_um[2]) / self.dx_um)
        return i, j, k

    def voxel_volume_um3(self) -> float:
        return float(self.dx_um ** 3)

    def voxel_volume_L(self) -> float:
        # 1 µm^3 = 1e-15 L
        return float(self.voxel_volume_um3() * 1e-15)

    def liquid_volume_L(self) -> float:
        # "total liquid volume of the global space" — exclude solids and (coarse) hypha voxels
        liquid_mask = (~self.solid_mask) & (~self.hypha_mask)
        return float(np.sum(liquid_mask) * self.voxel_volume_L())


# =============================================================================
# NUTRIENT INIT
# =============================================================================

def init_gaussian_field(n, mean, sigma, clip_half_range=None, rng=None):
    field = mean + rng.normal(0.0, sigma, size=(n, n, n))
    if clip_half_range is not None:
        lo, hi = mean - clip_half_range, mean + clip_half_range
        field = np.clip(field, lo, hi)
    field = field - field.mean() + mean
    if clip_half_range is not None:
        field = np.clip(field, lo, hi)
        field = field - field.mean() + mean
    return field.astype(np.float32)

def apply_solid_mask_to_nutrients(grid: VoxelGrid3D):
    for name in grid.nutrients:
        arr = grid.nutrients[name]
        arr[grid.solid_mask] = np.nan
        grid.nutrients[name] = arr

def initialize_nutrients(grid: VoxelGrid3D):
    rng = np.random.default_rng(RNG_SEED)
    for name, spec in NUTRIENT_SPECS.items():
        mean = float(spec["mean"])
        pm = float(spec["pm"])
        sigma = pm * float(SIGMA_SCALE[name])
        clip = pm if CLAMP_TO_PM_RANGE else None
        grid.nutrients[name] = init_gaussian_field(grid.n, mean, sigma, clip, rng=rng)
    apply_solid_mask_to_nutrients(grid)


# =============================================================================
# CONIDIA + STARTING HYPHA
# =============================================================================

def random_unit_vector(rng):
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-12)

def sample_point_in_octant(rng, grid: VoxelGrid3D, radius_um, octant_signs):
    sx, sy, _sz = octant_signs

    outer_min = grid.min_um + radius_um
    outer_max = grid.max_um - radius_um

    half_inner = CONIDIA_SPAWN_BOX_EXTENT_UM / 2.0
    inner_min = grid.origin_um - half_inner
    inner_max = grid.origin_um + half_inner

    xmin, ymin, _zmin = np.maximum(inner_min, outer_min)
    xmax, ymax, _zmax = np.minimum(inner_max, outer_max)

    if xmin >= xmax or ymin >= ymax:
        xmin, ymin, _zmin = outer_min
        xmax, ymax, _zmax = outer_max

    ox, oy, oz = grid.origin_um
    z0 = float(np.clip(oz, outer_min[2], outer_max[2]))

    if sx > 0:
        xr = (max(ox, xmin), xmax)
    else:
        xr = (xmin, min(ox, xmax))

    if sy > 0:
        yr = (max(oy, ymin), ymax)
    else:
        yr = (ymin, min(oy, ymax))

    if xr[0] >= xr[1]:
        xr = (xmin, xmax)
    if yr[0] >= yr[1]:
        yr = (ymin, ymax)

    return np.array([rng.uniform(*xr), rng.uniform(*yr), z0], dtype=float)

def place_conidia(grid: VoxelGrid3D, n_conidia, radius_um, min_sep_um, seed):
    rng = np.random.default_rng(seed)
    octants = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    rng.shuffle(octants)

    centers = []
    tries = 0
    while len(centers) < n_conidia and tries < MAX_PLACEMENT_TRIES:
        tries += 1
        octant = octants[len(centers)] if len(centers) < len(octants) else octants[rng.integers(0, 8)]
        p = sample_point_in_octant(rng, grid, radius_um, octant)
        if all(np.linalg.norm(p - c) >= min_sep_um for c in centers):
            centers.append(p)

    if len(centers) < n_conidia:
        print(f"Warning: placed {len(centers)}/{n_conidia} conidia (reduce MIN_CONIDIA_SEPARATION_UM).")

    return np.array(centers, dtype=float)

def carve_sphere_from_liquid(grid: VoxelGrid3D, center_um, radius_um):
    Xc, Yc, Zc = grid.rebuild_center_mesh()
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    inside = (dx*dx + dy*dy + dz*dz) <= (radius_um * radius_um)
    grid.solid_mask |= inside
    apply_solid_mask_to_nutrients(grid)

def assign_starting_hyphae(conidia_centers_um, conidium_radius_um, seed):
    rng = np.random.default_rng(seed)
    n = len(conidia_centers_um)
    p_miss = MISS_HYPHA_BASE_PROB + MISS_HYPHA_PER_CONIDIUM * max(0, n - 1)
    p_miss = float(np.clip(p_miss, 0.0, MISS_HYPHA_MAX_PROB))

    starts = []
    for cid, center in enumerate(conidia_centers_um):
        if rng.random() < p_miss:
            starts.append({"conidium_id": cid, "has_hypha": False,
                           "start_point_um": None, "direction_unit": None})
            continue
        normal = random_unit_vector(rng)
        start_point = center + conidium_radius_um * normal
        starts.append({"conidium_id": cid, "has_hypha": True,
                       "start_point_um": start_point, "direction_unit": normal})
    return starts


# =============================================================================
# SKELETON GRAPHS
# =============================================================================

def initialize_hypha_graphs(grid: VoxelGrid3D, hypha_starts):
    n_conidia = len(hypha_starts)
    grid.metabolites["citric_acid_total_mg"] = np.zeros(n_conidia, dtype=np.float64)

    graphs = {cid: nx.Graph() for cid in range(n_conidia)}
    start_dirs = {}

    for item in hypha_starts:
        cid = item["conidium_id"]
        if not item["has_hypha"]:
            continue

        p0 = np.array(item["start_point_um"], dtype=float)
        graphs[cid].add_node(
            0,
            pos=p0,
            birth_step=0,
            branch_clock_step=0,
            branch_origin="root",
        )
        start_dirs[cid] = np.array(item["direction_unit"], dtype=float)

        idx = grid.world_to_index(*p0)
        if idx is not None:
            i, j, k = idx
            grid.hypha_mask[i, j, k] = True
            grid.hypha_owner[i, j, k] = cid

    return graphs, start_dirs

def node_pos(G: nx.Graph, n):
    return np.array(G.nodes[n]["pos"], dtype=float)

def total_edge_length_um(G: nx.Graph):
    L = 0.0
    for u, v in G.edges:
        pu = node_pos(G, u)
        pv = node_pos(G, v)
        L += float(np.linalg.norm(pv - pu))
    return L

def count_branch_nodes(G: nx.Graph):
    return sum(1 for n in G.nodes if G.degree[n] >= 3)

def get_growing_tips(G: nx.Graph, root=0):
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() == 1:
        return [next(iter(G.nodes))]
    tips = []
    for n in G.nodes:
        if n == root:
            continue
        if G.degree[n] == 1:
            tips.append(n)
    return tips

def count_tips(G: nx.Graph, root=0):
    return len(get_growing_tips(G, root=root))


# =============================================================================
# SENSING / DIRECTION / VARIABLE STEP
# =============================================================================

def _infer_tip_direction(tip_um, prev_um, default_dir_unit):
    if prev_um is not None:
        v = tip_um - prev_um
        n = np.linalg.norm(v)
        if n > 1e-12:
            return v / n
    d = np.array(default_dir_unit, dtype=float)
    return d / (np.linalg.norm(d) + 1e-12)

def _surface_touching_voxels(grid: VoxelGrid3D, center_um, radius_um):
    Xc, Yc, Zc = grid.rebuild_center_mesh()
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    shell_half_thickness = grid.dx_um * SURFACE_SHELL_THICKNESS_MULT
    shell = np.abs(dist - radius_um) <= shell_half_thickness
    return shell

def _candidate_pack(grid: VoxelGrid3D, conidium_id: int, center_um: np.ndarray):
    shell_mask = _surface_touching_voxels(grid, center_um, HYPHA_RADIUS_UM)
    shell_liquid = shell_mask & (~grid.solid_mask) & (~grid.hypha_mask)

    shell_has_any_hypha = shell_mask & grid.hypha_mask
    any_hypha_present = bool(np.any(shell_has_any_hypha))

    cand_idx = np.argwhere(shell_liquid)
    if cand_idx.size == 0:
        return any_hypha_present, None

    N = grid.nutrients["Nitrogen"]
    P = grid.nutrients["Phosphorus"]

    cand_N = N[cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]
    cand_P = P[cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]
    finite = np.isfinite(cand_N) & np.isfinite(cand_P)

    cand_idx = cand_idx[finite]
    cand_N = cand_N[finite]
    cand_P = cand_P[finite]

    if cand_idx.size == 0:
        return any_hypha_present, None

    return any_hypha_present, (cand_idx, cand_N.astype(float), cand_P.astype(float))

def _choose_best_direction(grid: VoxelGrid3D, node_um: np.ndarray, traj_dir: np.ndarray, cand_pack):
    cand_idx, cand_N, cand_P = cand_pack
    score = cand_N * cand_P

    vx = grid.x_centers[cand_idx[:, 0]]
    vy = grid.y_centers[cand_idx[:, 1]]
    vz = grid.z_centers[cand_idx[:, 2]]
    vecs = np.stack([vx, vy, vz], axis=1) - node_um
    vecs_norm = np.linalg.norm(vecs, axis=1) + 1e-12
    dirs = vecs / vecs_norm[:, None]

    forward = np.clip(np.einsum("ij,j->i", dirs, traj_dir), -1.0, 1.0)
    blended = score * (1.0 + TRAJECTORY_BIAS * forward)

    best = int(np.argmax(blended))
    target_ijk = tuple(int(x) for x in cand_idx[best])
    grow_dir = dirs[best]
    n_at = float(cand_N[best])
    p_at = float(cand_P[best])
    return target_ijk, grow_dir, n_at, p_at

def _growth_step_from_np(n_val: float, p_val: float) -> float:
    n_set = float(NUTRIENT_SPECS["Nitrogen"]["mean"])
    p_set = float(NUTRIENT_SPECS["Phosphorus"]["mean"])
    mult = 0.5 * ((n_val / (n_set + 1e-12)) + (p_val / (p_set + 1e-12)))
    mult = float(np.clip(mult, MIN_GROWTH_MULT, MAX_GROWTH_MULT))
    return float(GROWTH_STEP_UM * mult)

def node_starved_at_own_voxel(grid: VoxelGrid3D, node_um: np.ndarray) -> bool:
    idx = grid.world_to_index(*node_um)
    if idx is None:
        return False
    i, j, k = idx
    if grid.hypha_mask[i, j, k]:
        return False
    nval = grid.nutrients["Nitrogen"][i, j, k]
    pval = grid.nutrients["Phosphorus"][i, j, k]
    if not (np.isfinite(nval) and np.isfinite(pval)):
        return False
    return (nval < N_GROWTH_THRESHOLD) or (pval < P_GROWTH_THRESHOLD)


# =============================================================================
# BRANCH DIRECTION HELPERS
# =============================================================================

def _random_perp_basis(d: np.ndarray, rng: np.random.Generator):
    d = d / (np.linalg.norm(d) + 1e-12)
    a = random_unit_vector(rng)
    u = a - d * np.dot(a, d)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(d, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v

def _make_apical_dirs(base_dir: np.ndarray, rng: np.random.Generator):
    base = base_dir / (np.linalg.norm(base_dir) + 1e-12)
    u, _v = _random_perp_basis(base, rng)
    theta = np.deg2rad(rng.uniform(APICAL_BRANCH_ANGLE_DEG_MIN, APICAL_BRANCH_ANGLE_DEG_MAX))
    half = theta / 2.0
    d1 = np.cos(half) * base + np.sin(half) * u
    d2 = np.cos(half) * base - np.sin(half) * u
    d1 /= (np.linalg.norm(d1) + 1e-12)
    d2 /= (np.linalg.norm(d2) + 1e-12)
    return d1, d2

def _is_position_valid_for_new_node(grid: VoxelGrid3D, cid: int, pos_um: np.ndarray) -> bool:
    idx = grid.world_to_index(*pos_um)
    if idx is None:
        return False
    i, j, k = idx
    if grid.solid_mask[i, j, k]:
        return False
    if grid.hypha_mask[i, j, k]:
        return False
    return True


# =============================================================================
# check_local_conditions (BRANCH CLOCK FIXED)
# =============================================================================

def check_local_conditions(grid: VoxelGrid3D,
                           cid: int,
                           node_um: np.ndarray,
                           prev_um: np.ndarray | None,
                           first_node_default_dir_unit: np.ndarray | None,
                           step_idx: int,
                           branch_clock_step: int,
                           rng: np.random.Generator):

    node_um = np.array(node_um, dtype=float)

    if node_starved_at_own_voxel(grid, node_um):
        return {
            "action": "SECRETE_ONLY",
            "reason": "LOCAL_NODE_STARVED",
            "conidium_id": cid,
            "node_um": node_um,
            "citric_acid_g_per_L": CITRIC_ACID_RATE_G_PER_L_PER_STEP,  # kept (ignored)
            "deplete_mode": "SECRETE",
        }

    if first_node_default_dir_unit is None and prev_um is None:
        raise ValueError("Need first_node_default_dir_unit when prev_um is None.")

    prev_um = None if prev_um is None else np.array(prev_um, dtype=float)
    traj_dir = _infer_tip_direction(node_um, prev_um, first_node_default_dir_unit)

    any_hypha_present, cand_pack = _candidate_pack(grid, cid, node_um)
    if cand_pack is None:
        return {
            "action": "SECRETE_ONLY",
            "reason": "NO_LIQUID_CONTACT",
            "conidium_id": cid,
            "node_um": node_um,
            "citric_acid_g_per_L": CITRIC_ACID_RATE_G_PER_L_PER_STEP,  # kept (ignored)
            "deplete_mode": "SECRETE",
        }

    cand_idx, cand_N, cand_P = cand_pack
    low_in_all = np.all((cand_N < N_GROWTH_THRESHOLD) | (cand_P < P_GROWTH_THRESHOLD))
    if low_in_all:
        return {
            "action": "SECRETE_ONLY",
            "reason": "LOW_N_OR_P_EVERYWHERE",
            "conidium_id": cid,
            "node_um": node_um,
            "citric_acid_g_per_L": CITRIC_ACID_RATE_G_PER_L_PER_STEP,  # kept (ignored)
            "deplete_mode": "SECRETE",
        }

    target_ijk, grow_dir, n_at, p_at = _choose_best_direction(grid, node_um, traj_dir, cand_pack)
    step_um = _growth_step_from_np(n_at, p_at)

    age = step_idx - int(branch_clock_step)
    if age >= APICAL_BRANCH_MIN_AGE_STEPS and (n_at >= APICAL_N_THRESHOLD) and (p_at >= APICAL_P_THRESHOLD):
        if rng.random() < APICAL_BRANCH_PROB:
            for _ in range(60):
                d1, d2 = _make_apical_dirs(grow_dir, rng)
                p1 = node_um + step_um * d1
                p2 = node_um + step_um * d2
                if _is_position_valid_for_new_node(grid, cid, p1) and _is_position_valid_for_new_node(grid, cid, p2):
                    return {
                        "action": "APICAL_BRANCH",
                        "reason": "APICAL_BRANCH",
                        "conidium_id": cid,
                        "node_um": node_um,
                        "branch_dirs_unit": (d1, d2),
                        "branch_steps_um": (step_um, step_um),
                        "deplete_mode": "GROW",
                        "n_at": n_at,
                        "p_at": p_at,
                        "any_hypha_present": any_hypha_present,
                    }

    return {
        "action": "GROW",
        "reason": "GROW",
        "conidium_id": cid,
        "node_um": node_um,
        "grow_dir_unit": grow_dir,
        "growth_step_um": step_um,
        "target_voxel_ijk": target_ijk,
        "deplete_mode": "GROW",
        "n_at": n_at,
        "p_at": p_at,
        "any_hypha_present": any_hypha_present,
    }


# =============================================================================
# decision(...) enacts GROW / APICAL_BRANCH / SECRETE_ONLY  (BRANCH CLOCK FIXED)
# =============================================================================

def _add_node_and_edge(grid, G, cid, parent, new_pos, step_idx, origin, inherit_branch_clock_step):
    new_id = (max(G.nodes) + 1) if G.number_of_nodes() > 0 else 0
    G.add_node(
        new_id,
        pos=np.array(new_pos, dtype=float),
        birth_step=step_idx,
        branch_clock_step=int(inherit_branch_clock_step),
        branch_origin=origin,
    )
    G.add_edge(parent, new_id)

    idx = grid.world_to_index(*new_pos)
    if idx is not None:
        i, j, k = idx
        grid.hypha_mask[i, j, k] = True
        grid.hypha_owner[i, j, k] = cid
        for nutr in grid.nutrients:
            if not grid.solid_mask[i, j, k] and np.isfinite(grid.nutrients[nutr][i, j, k]):
                grid.nutrients[nutr][i, j, k] = 0.0

    return new_id, idx

def decision(grid: VoxelGrid3D,
             G: nx.Graph,
             cid: int,
             node_id: int,
             dec: dict,
             step_idx: int,
             branch_event_counters: dict):
    action = dec["action"]

    # IMPORTANT CHANGE:
    # We no longer add a predefined citric amount here.
    # Citric acid is produced from glucose consumed in apply_action_local_depletion().
    if action == "SECRETE_ONLY":
        return {"did": "SECRETE_ONLY"}, "SECRETE"

    if action == "GROW":
        tip = np.array(dec["node_um"], dtype=float)
        d = np.array(dec["grow_dir_unit"], dtype=float)
        d = d / (np.linalg.norm(d) + 1e-12)
        step_um = float(dec.get("growth_step_um", GROWTH_STEP_UM))
        new_pos = tip + step_um * d

        if not _is_position_valid_for_new_node(grid, cid, new_pos):
            return {"did": "SECRETE_ONLY", "reason": "GROW_BLOCKED"}, "SECRETE"

        parent_clock = int(G.nodes[node_id].get("branch_clock_step", 0))
        new_id, idx = _add_node_and_edge(
            grid, G, cid, node_id, new_pos, step_idx,
            origin="grow",
            inherit_branch_clock_step=parent_clock
        )
        return {"did": "GROW", "new_node": new_id, "marked_voxel": idx, "step_um": step_um}, "GROW"

    if action == "APICAL_BRANCH":
        tip = np.array(dec["node_um"], dtype=float)
        (d1, d2) = dec["branch_dirs_unit"]
        (s1, s2) = dec["branch_steps_um"]

        p1 = tip + float(s1) * np.array(d1, dtype=float)
        p2 = tip + float(s2) * np.array(d2, dtype=float)

        ok1 = _is_position_valid_for_new_node(grid, cid, p1)
        ok2 = _is_position_valid_for_new_node(grid, cid, p2)

        if ok1 and ok2:
            new_clock = step_idx
            n1, idx1 = _add_node_and_edge(grid, G, cid, node_id, p1, step_idx, origin="apical",
                                          inherit_branch_clock_step=new_clock)
            n2, idx2 = _add_node_and_edge(grid, G, cid, node_id, p2, step_idx, origin="apical",
                                          inherit_branch_clock_step=new_clock)
            branch_event_counters["apical"] += 1
            return {"did": "APICAL_BRANCH", "new_nodes": (n1, n2), "voxels": (idx1, idx2)}, "GROW"

        if ok1:
            parent_clock = int(G.nodes[node_id].get("branch_clock_step", 0))
            n1, idx1 = _add_node_and_edge(grid, G, cid, node_id, p1, step_idx, origin="grow",
                                          inherit_branch_clock_step=parent_clock)
            return {"did": "GROW", "new_node": n1, "marked_voxel": idx1, "fallback": "apical_to_grow"}, "GROW"

        return {"did": "SECRETE_ONLY", "reason": "APICAL_BLOCKED"}, "SECRETE"

    raise ValueError(f"Unknown action: {action}")


# =============================================================================
# LATERAL BRANCHING
# =============================================================================

def _multi_source_dist_to_tips(G: nx.Graph, tips: list[int]) -> dict[int, int]:
    dist = {n: 10**9 for n in G.nodes}
    q = []
    for t in tips:
        dist[t] = 0
        q.append(t)
    head = 0
    while head < len(q):
        u = q[head]
        head += 1
        for v in G.neighbors(u):
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def lateral_branch_attempts(grid: VoxelGrid3D,
                            G: nx.Graph,
                            cid: int,
                            start_dir: np.ndarray,
                            step_idx: int,
                            rng: np.random.Generator,
                            branch_event_counters: dict):
    outcomes = []
    if G.number_of_nodes() < 2:
        return outcomes

    tips = get_growing_tips(G, root=0)
    if len(tips) == 0:
        return outcomes

    dist_to_tip = _multi_source_dist_to_tips(G, tips)

    body_nodes = []
    for n in G.nodes:
        if n == 0:
            continue
        if n in tips:
            continue
        if dist_to_tip.get(n, 0) >= LATERAL_MIN_DISTANCE_FROM_TIP:
            body_nodes.append(n)

    if len(body_nodes) == 0:
        return outcomes

    rng.shuffle(body_nodes)
    for n in body_nodes:
        if rng.random() > LATERAL_BRANCH_PROB:
            continue

        pos = node_pos(G, n)

        if node_starved_at_own_voxel(grid, pos):
            dec = {
                "action": "SECRETE_ONLY",
                "conidium_id": cid,
                "node_um": pos,
                "citric_acid_g_per_L": CITRIC_ACID_RATE_G_PER_L_PER_STEP,  # ignored
                "deplete_mode": "SECRETE",
            }
            out, tag = decision(grid, G, cid, n, dec, step_idx, branch_event_counters)
            outcomes.append((n, out, tag))
            continue

        _, cand_pack = _candidate_pack(grid, cid, pos)
        if cand_pack is None:
            continue

        cand_idx, cand_N, cand_P = cand_pack
        good = (cand_N >= LATERAL_N_THRESHOLD) & (cand_P >= LATERAL_P_THRESHOLD)
        if not np.any(good):
            continue

        gi = np.where(good)[0]
        best_local = gi[int(np.argmax((cand_N[gi] * cand_P[gi])))]
        target_idx = cand_idx[best_local]
        n_at = float(cand_N[best_local])
        p_at = float(cand_P[best_local])

        vx = grid.x_centers[target_idx[0]]
        vy = grid.y_centers[target_idx[1]]
        vz = grid.z_centers[target_idx[2]]
        vec = np.array([vx, vy, vz], dtype=float) - pos
        vec /= (np.linalg.norm(vec) + 1e-12)

        step_um = _growth_step_from_np(n_at, p_at)
        new_pos = pos + step_um * vec

        if not _is_position_valid_for_new_node(grid, cid, new_pos):
            continue

        new_id, idx = _add_node_and_edge(
            grid, G, cid, n, new_pos, step_idx,
            origin="lateral",
            inherit_branch_clock_step=step_idx
        )
        branch_event_counters["lateral"] += 1
        outcomes.append((n, {"did": "LATERAL_BRANCH", "new_node": new_id, "marked_voxel": idx}, "GROW"))
        break

    return outcomes


# =============================================================================
# ACTION-LOCAL DEPLETION AROUND ACTIVE NODES
# =============================================================================

def _neighbors_26(i, j, k, n):
    for di in (-1, 0, 1):
        ii = i + di
        if ii < 0 or ii >= n:
            continue
        for dj in (-1, 0, 1):
            jj = j + dj
            if jj < 0 or jj >= n:
                continue
            for dk in (-1, 0, 1):
                kk = k + dk
                if kk < 0 or kk >= n:
                    continue
                if di == 0 and dj == 0 and dk == 0:
                    continue
                yield ii, jj, kk

def _compute_glucose_consumed_g_for_node(grid: VoxelGrid3D, i: int, j: int, k: int, factor: float) -> float:
    """
    Computes how many grams of glucose are consumed in the 26-neighborhood
    by applying the same multiplicative depletion rule.

    Consumption per voxel (g) = (conc_g_per_L * voxel_volume_L) * (1 - factor)
    where conc is the *current* glucose concentration before depletion.
    """
    if factor >= 1.0:
        return 0.0

    vol_L = grid.voxel_volume_L()
    Glc = grid.nutrients["Glucose"]
    n = grid.n

    consumed_g = 0.0
    for ii, jj, kk in _neighbors_26(i, j, k, n):
        if grid.solid_mask[ii, jj, kk] or grid.hypha_mask[ii, jj, kk]:
            continue
        c = Glc[ii, jj, kk]
        if not np.isfinite(c):
            continue
        mass_g = float(c) * vol_L
        consumed_g += mass_g * (1.0 - float(factor))

    return float(consumed_g)

def apply_action_local_depletion(grid: VoxelGrid3D,
                                 graphs: dict[int, nx.Graph],
                                 per_node_actions: dict[tuple[int, int], str],
                                 use_26_neighbors=True):
    neigh_fn = _neighbors_26 if use_26_neighbors else None
    n = grid.n

    for (cid, node_id), tag in per_node_actions.items():
        G = graphs[cid]
        if node_id not in G.nodes:
            continue

        pos = node_pos(G, node_id)
        idx = grid.world_to_index(*pos)
        if idx is None:
            continue
        i, j, k = idx

        if G.number_of_nodes() == 1:
            factor = DEPLETION_TIP_FACTOR
        else:
            if node_id != 0 and G.degree[node_id] == 1:
                factor = DEPLETION_TIP_FACTOR
            elif G.degree[node_id] == 2:
                factor = DEPLETION_LINK_FACTOR
            else:
                factor = DEPLETION_LINK_FACTOR

        if tag == "GROW":
            to_deplete = {"Nitrogen", "Phosphorus"}
        else:
            to_deplete = {"Glucose"}

            # NEW: compute glucose consumed (grams) in surrounding voxels,
            # convert to mg citric acid and accumulate.
            glucose_consumed_g = _compute_glucose_consumed_g_for_node(grid, i, j, k, factor=factor)
            citric_mg = glucose_consumed_g * CITRIC_MG_PER_G_GLUCOSE
            grid.metabolites["citric_acid_total_mg"][cid] += citric_mg
            
            # Also accumulate spatially for each neighbor voxel
            vol_L = grid.voxel_volume_L()
            Glc = grid.nutrients["Glucose"]
            for ii, jj, kk in neigh_fn(i, j, k, n):
                if grid.solid_mask[ii, jj, kk] or grid.hypha_mask[ii, jj, kk]:
                    continue
                c = Glc[ii, jj, kk]
                if not np.isfinite(c):
                    continue
                mass_g = float(c) * vol_L
                consumed_g_voxel = mass_g * (1.0 - float(factor))
                citric_mg_voxel = consumed_g_voxel * CITRIC_MG_PER_G_GLUCOSE
                grid.metabolites["citric_acid_spatial_mg"][ii, jj, kk] += citric_mg_voxel

        for ii, jj, kk in neigh_fn(i, j, k, n):
            if grid.solid_mask[ii, jj, kk] or grid.hypha_mask[ii, jj, kk]:
                continue
            for nutr in to_deplete:
                arr = grid.nutrients[nutr]
                if np.isfinite(arr[ii, jj, kk]):
                    arr[ii, jj, kk] = arr[ii, jj, kk] * factor
                    grid.nutrients[nutr] = arr

def mean_nutrients_over_liquid(grid: VoxelGrid3D):
    liquid_mask = (~grid.solid_mask) & (~grid.hypha_mask)
    means = {}
    for nutr, arr in grid.nutrients.items():
        m = liquid_mask & np.isfinite(arr)
        means[nutr] = float(np.mean(arr[m])) if np.any(m) else float("nan")
    return means


# =============================================================================
# DIFFUSION
# =============================================================================

def diffuse_nutrients(grid: VoxelGrid3D, seed=0):
    rng = np.random.default_rng(seed)
    liquid_mask = (~grid.solid_mask) & (~grid.hypha_mask)

    shifts = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    for nutr, arr in grid.nutrients.items():
        rate = float(DIFFUSION_RATE.get(nutr, 0.0))
        if rate <= 0:
            continue

        valid = liquid_mask & np.isfinite(arr)
        arr0 = np.where(valid, arr, 0.0).astype(np.float32)
        cnt0 = valid.astype(np.float32)

        neigh_sum = np.zeros_like(arr0)
        neigh_cnt = np.zeros_like(cnt0)

        for di, dj, dk in shifts:
            neigh_sum += np.roll(arr0, shift=(di, dj, dk), axis=(0,1,2))
            neigh_cnt += np.roll(cnt0, shift=(di, dj, dk), axis=(0,1,2))

        avg = np.where(neigh_cnt > 0, neigh_sum / (neigh_cnt + 1e-12), arr0)

        new_arr = arr.copy()
        delta = rate * (avg - arr0)

        noise = (rng.random(size=arr0.shape).astype(np.float32) - 0.5) * 2.0
        base_scale = np.nanmean(arr0[valid]) if np.any(valid) else 0.0
        noise_term = DIFFUSION_NOISE_FRACTION * np.where(arr0 > 0, arr0, base_scale) * noise

        upd = delta + noise_term
        new_arr[valid] = np.maximum(0.0, (arr0[valid] + upd[valid]))

        new_arr[grid.hypha_mask] = 0.0
        new_arr[grid.solid_mask] = np.nan
        grid.nutrients[nutr] = new_arr.astype(np.float32)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(grid: VoxelGrid3D, graphs: dict[int, nx.Graph]):
    total_length = 0.0
    total_branch_nodes = 0
    total_tips = 0

    for cid, G in graphs.items():
        total_length += total_edge_length_um(G)
        total_branch_nodes += count_branch_nodes(G)
        total_tips += count_tips(G, root=0)

    total_volume = total_length * np.pi * (HYPHA_RADIUS_UM ** 2)

    # NEW: compute mg/L from cumulative mg and current liquid volume
    citric_total_mg = float(np.sum(grid.metabolites["citric_acid_total_mg"]))
    liquid_L = grid.liquid_volume_L()
    citric_mg_per_L = (citric_total_mg / liquid_L) if (liquid_L > 0) else float("nan")

    return {
        "total_length_um": total_length,
        "total_volume_um3": total_volume,
        "total_branch_nodes": total_branch_nodes,
        "total_tips": total_tips,
        "citric_total_mg": citric_total_mg,
        "citric_mg_per_L": citric_mg_per_L,
        "liquid_volume_L": liquid_L,
    }


# =============================================================================
# NEW PLOTS + CSV EXPORTS  (unchanged)
# =============================================================================

def plot_axes_and_grid(grid: VoxelGrid3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    half = EXTENT_UM / 2.0
    ax.plot([-half, half], [0, 0], [0, 0], linewidth=2)
    ax.plot([0, 0], [-half, half], [0, 0], linewidth=2)
    ax.plot([0, 0], [0, 0], [-half, half], linewidth=2)

    stride = max(1, int(GRID_PLOT_STRIDE))
    xs = grid.x_edges[::stride]
    ys = grid.y_edges[::stride]
    zs = grid.z_edges[::stride]

    seg_rows = []

    for y in ys:
        for z in zs:
            x0, x1 = grid.min_um[0], grid.max_um[0]
            ax.plot([x0, x1], [y, y], [z, z], linewidth=0.5, alpha=0.25)
            seg_rows.append({"x0": x0, "y0": y, "z0": z, "x1": x1, "y1": y, "z1": z})

    for x in xs:
        for z in zs:
            y0, y1 = grid.min_um[1], grid.max_um[1]
            ax.plot([x, x], [y0, y1], [z, z], linewidth=0.5, alpha=0.25)
            seg_rows.append({"x0": x, "y0": y0, "z0": z, "x1": x, "y1": y1, "z1": z})

    for x in xs:
        for y in ys:
            z0, z1 = grid.min_um[2], grid.max_um[2]
            ax.plot([x, x], [y, y], [z0, z1], linewidth=0.5, alpha=0.25)
            seg_rows.append({"x0": x, "y0": y, "z0": z0, "x1": x, "y1": y, "z1": z1})

    ax.set_title("Cartesian axes + 3D voxel grid (subsampled)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_xlim(grid.min_um[0], grid.max_um[0])
    ax.set_ylim(grid.min_um[1], grid.max_um[1])
    ax.set_zlim(grid.min_um[2], grid.max_um[2])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    write_csv(
        "plot_axes_grid_segments.csv",
        fieldnames=["x0","y0","z0","x1","y1","z1"],
        rows=seg_rows
    )

def plot_spores_and_initial_dirs(conidia_centers_um, hypha_starts, conidium_radius_um):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    rows = []
    for item in hypha_starts:
        cid = item["conidium_id"]
        center = conidia_centers_um[cid]
        ax.scatter([center[0]], [center[1]], [center[2]], s=80)

        row = {
            "conidium_id": cid,
            "center_x_um": float(center[0]),
            "center_y_um": float(center[1]),
            "center_z_um": float(center[2]),
            "has_hypha": bool(item["has_hypha"]),
        }

        if item["has_hypha"]:
            p0 = np.array(item["start_point_um"], dtype=float)
            d = np.array(item["direction_unit"], dtype=float)
            d = d / (np.linalg.norm(d) + 1e-12)

            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=30)
            arrow_len = max(5.0, conidium_radius_um * 2.0)
            ax.quiver(p0[0], p0[1], p0[2], d[0], d[1], d[2], length=arrow_len, normalize=True)

            row.update({
                "start_x_um": float(p0[0]),
                "start_y_um": float(p0[1]),
                "start_z_um": float(p0[2]),
                "dir_x": float(d[0]),
                "dir_y": float(d[1]),
                "dir_z": float(d[2]),
            })
        else:
            row.update({
                "start_x_um": "",
                "start_y_um": "",
                "start_z_um": "",
                "dir_x": "",
                "dir_y": "",
                "dir_z": "",
            })

        rows.append(row)

    ax.set_title("Spore locations + initial hypha direction(s)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    half = EXTENT_UM / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    write_csv(
        "plot_spores_initial_dirs.csv",
        fieldnames=["conidium_id","center_x_um","center_y_um","center_z_um","has_hypha",
                    "start_x_um","start_y_um","start_z_um","dir_x","dir_y","dir_z"],
        rows=rows
    )

def plot_skeletons_3d_with_node_age(graphs: dict[int, nx.Graph], grid: VoxelGrid3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for cid, G in graphs.items():
        if G.number_of_nodes() == 0:
            continue

        node_rows = []
        edge_rows = []

        birth = np.array([int(G.nodes[n].get("birth_step", 0)) for n in G.nodes], dtype=float)
        bmin, bmax = float(np.min(birth)), float(np.max(birth))
        denom = (bmax - bmin) if (bmax > bmin) else 1.0

        cmap = cm.get_cmap("RdYlGn")

        for u, v in G.edges:
            pu = node_pos(G, u); pv = node_pos(G, v)
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]], alpha=0.35, linewidth=1.5)
            edge_rows.append({
                "u": int(u), "v": int(v),
                "ux_um": float(pu[0]), "uy_um": float(pu[1]), "uz_um": float(pu[2]),
                "vx_um": float(pv[0]), "vy_um": float(pv[1]), "vz_um": float(pv[2]),
            })

        for n in G.nodes:
            p = node_pos(G, n)
            bs = float(G.nodes[n].get("birth_step", 0))
            t = (bs - bmin) / denom
            col = cmap(t)
            ax.scatter([p[0]], [p[1]], [p[2]], s=18, color=col)
            node_rows.append({
                "node_id": int(n),
                "x_um": float(p[0]),
                "y_um": float(p[1]),
                "z_um": float(p[2]),
                "birth_step": int(bs),
                "birth_time_hr": float(bs * TIME_STEP_HR),
                "degree": int(G.degree[n]),
                "branch_origin": str(G.nodes[n].get("branch_origin", "")),
                "branch_clock_step": int(G.nodes[n].get("branch_clock_step", 0)),
            })

        if 0 in G.nodes:
            p0 = node_pos(G, 0)
            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=90, marker="*", color="k")
            ax.text(p0[0], p0[1], p0[2], f"  cid{cid} start", fontsize=9)

        write_csv(
            f"skeleton_nodes_cid{cid}.csv",
            fieldnames=["node_id","x_um","y_um","z_um","birth_step","birth_time_hr","degree","branch_origin","branch_clock_step"],
            rows=node_rows
        )
        write_csv(
            f"skeleton_edges_cid{cid}.csv",
            fieldnames=["u","v","ux_um","uy_um","uz_um","vx_um","vy_um","vz_um"],
            rows=edge_rows
        )

    ax.set_title("Final hypha skeletons (nodes colored red→green by age)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (µm)")
    ax.set_xlim(grid.min_um[0], grid.max_um[0])
    ax.set_ylim(grid.min_um[1], grid.max_um[1])
    ax.set_zlim(grid.min_um[2], grid.max_um[2])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

def export_time_series_csv(history):
    _ensure_outdir()
    n = len(history["total_length_um"])
    rows = []
    for i in range(n):
        row = {
            "step": i,
            "time_hr": float(i * TIME_STEP_HR),
            "total_length_um": float(history["total_length_um"][i]),
            "total_volume_um3": float(history["total_volume_um3"][i]),
            "total_branch_nodes": int(history["total_branch_nodes"][i]),
            "total_tips": int(history["total_tips"][i]),
            # NEW:
            "citric_total_mg": float(history["citric_total_mg"][i]),
            "citric_mg_per_L": float(history["citric_mg_per_L"][i]),
            "liquid_volume_L": float(history["liquid_volume_L"][i]),
            "apical_branch_events": int(history["apical_branch_events"][i]),
            "lateral_branch_events": int(history["lateral_branch_events"][i]),
        }
        for nutr in NUTRIENT_SPECS.keys():
            row[f"mean_{nutr}_g_per_L"] = float(history["mean_nutrients"][nutr][i])
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["step","time_hr"]
    write_csv("time_series_all.csv", fieldnames=fieldnames, rows=rows)

def plot_time_series(history):
    export_time_series_csv(history)

    t_hr = np.arange(len(history["total_length_um"])) * TIME_STEP_HR

    def exponential_func(x, a, b, c):
        """Exponential function: y = a * exp(b * x) + c"""
        return a * np.exp(b * x) + c

    def simple_plot(y, title, ylabel, add_exp_fit=False):
        # Set figure size to match the image dimensions (width, height in inches)
        plt.figure(figsize=(8, 4.5))
        plt.plot(t_hr, y, marker="o", label="Data")
        
        if add_exp_fit:
            # Perform exponential regression
            try:
                # Initial parameter guesses
                y_arr = np.array(y)
                popt, pcov = curve_fit(exponential_func, t_hr, y_arr, 
                                       p0=[y_arr[0], 0.1, 0],
                                       maxfev=10000)
                a, b, c = popt
                
                # Calculate fitted values
                y_fit = exponential_func(t_hr, a, b, c)
                
                # Calculate R²
                ss_res = np.sum((y_arr - y_fit) ** 2)
                ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Plot the fitted curve
                plt.plot(t_hr, y_fit, 'r--', linewidth=2, 
                        label=f'Exp. fit: y = {a:.2f} * exp({b:.4f} * x) + {c:.2f}')
                
                # Add R² to the plot
                plt.text(0.95, 0.95, f'R² = {r_squared:.2f}', 
                        transform=plt.gca().transAxes,
                        fontsize=12, verticalalignment='top', 
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.legend(loc='upper left')
            except Exception as e:
                print(f"Warning: Could not fit exponential to {title}: {e}")
        
        plt.title(title)
        plt.xlabel("Time (hrs)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    simple_plot(history["total_length_um"], "Total length over time", "Length (µm)", add_exp_fit=True)
    simple_plot(history["total_volume_um3"], "Total hyphal volume over time", "Volume (µm³)", add_exp_fit=True)
    simple_plot(history["total_branch_nodes"], "Total branch nodes (deg≥3) over time", "Count")
    simple_plot(history["total_tips"], "Total growing tips over time", "Count")

    # NEW plot requested: mg/L per timestep (based on cumulative mg / current liquid volume)
    simple_plot(history["citric_mg_per_L"], "Citric acid concentration over time", "Citric acid (mg/L)")

    # Optional (useful for debugging): cumulative mg
    simple_plot(history["citric_total_mg"], "Cumulative citric acid produced over time", "Citric acid (mg)")

    simple_plot(history["apical_branch_events"], "Apical branching events over time", "Cumulative events")
    simple_plot(history["lateral_branch_events"], "Lateral branching events over time", "Cumulative events")

    for nutr in NUTRIENT_SPECS.keys():
        simple_plot(history["mean_nutrients"][nutr], f"Mean {nutr} in liquid over time", "g/L")


# =============================================================================
# HYPHAL DENSITY ANALYSIS (SHELL-BASED)
# =============================================================================

def compute_shell_boundaries(center_um, graphs, n_shells=15):
    """
    Find the furthest point from center across all graphs and create shell boundaries.
    Returns: (max_radius, shell_radii) where shell_radii has n_shells+1 values
    """
    max_dist = 0.0
    for G in graphs.values():
        for n in G.nodes:
            pos = node_pos(G, n)
            dist = float(np.linalg.norm(pos - center_um))
            max_dist = max(max_dist, dist)
    
    if max_dist == 0.0:
        max_dist = 1.0  # fallback
    
    shell_radii = np.linspace(0, max_dist, n_shells + 1)
    return max_dist, shell_radii


def edge_length_in_shell(p1, p2, center, r_inner, r_outer):
    """
    Calculate the length of edge segment (p1, p2) that falls within the spherical shell
    defined by radii r_inner and r_outer from center.
    """
    p1, p2, center = np.array(p1), np.array(p2), np.array(center)
    
    # Edge vector
    edge_vec = p2 - p1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return 0.0
    
    # Parametric representation: p(t) = p1 + t*(p2-p1), t in [0,1]
    # We need to find what portion of the edge is within [r_inner, r_outer]
    
    # Sample the edge at fine resolution to determine intersection
    n_samples = 100
    t_vals = np.linspace(0, 1, n_samples)
    sample_points = p1[np.newaxis, :] + t_vals[:, np.newaxis] * edge_vec[np.newaxis, :]
    
    # Distances from center
    dists = np.linalg.norm(sample_points - center[np.newaxis, :], axis=1)
    
    # Which samples are in the shell?
    in_shell = (dists >= r_inner) & (dists <= r_outer)
    
    # Fraction of edge in shell
    fraction_in_shell = np.sum(in_shell) / n_samples
    
    return float(edge_len * fraction_in_shell)


def analyze_hyphal_density_by_shell(center_um, graphs, n_shells=15):
    """
    Analyze hyphal density as a function of distance from center.
    Returns dictionary with shell data.
    """
    max_radius, shell_radii = compute_shell_boundaries(center_um, graphs, n_shells)
    
    results = {
        'shell_radii': shell_radii,
        'shell_inner': shell_radii[:-1],
        'shell_outer': shell_radii[1:],
        'shell_mid': (shell_radii[:-1] + shell_radii[1:]) / 2,
        'shell_volumes': [],
        'hyphal_lengths': [],
        'hyphal_volumes': [],
        'hyphal_fractions': [],
        'edges_per_shell': [],  # for visualization
    }
    
    # Calculate volumes and hyphal content for each shell
    for i in range(n_shells):
        r_inner = float(shell_radii[i])
        r_outer = float(shell_radii[i + 1])
        
        # Shell volume: (4/3)π(r_outer³ - r_inner³)
        shell_vol = (4.0/3.0) * np.pi * (r_outer**3 - r_inner**3)
        results['shell_volumes'].append(shell_vol)
        
        # Calculate hyphal length in this shell
        total_length_in_shell = 0.0
        edges_in_shell = []
        
        for cid, G in graphs.items():
            for u, v in G.edges:
                pu = node_pos(G, u)
                pv = node_pos(G, v)
                
                length_in_shell = edge_length_in_shell(pu, pv, center_um, r_inner, r_outer)
                if length_in_shell > 0:
                    total_length_in_shell += length_in_shell
                    edges_in_shell.append((cid, u, v, pu, pv, length_in_shell))
        
        results['hyphal_lengths'].append(total_length_in_shell)
        results['edges_per_shell'].append(edges_in_shell)
        
        # Hyphal volume in shell
        hyphal_vol = total_length_in_shell * np.pi * (HYPHA_RADIUS_UM ** 2)
        results['hyphal_volumes'].append(hyphal_vol)
        
        # Hyphal fraction
        hyphal_frac = (hyphal_vol / shell_vol) if shell_vol > 0 else 0.0
        results['hyphal_fractions'].append(hyphal_frac)
    
    return results


def analyze_nutrient_by_shell(center_um, grid, nutrient_before, nutrient_after, nutrient_name, shell_radii):
    """
    Analyze nutrient concentration changes by shell.
    
    Args:
        center_um: Center point for shells
        grid: VoxelGrid3D object
        nutrient_before: 3D array of nutrient concentrations before depletion
        nutrient_after: 3D array of nutrient concentrations after depletion
        nutrient_name: Name of the nutrient
        shell_radii: Array of shell boundary radii
    
    Returns:
        Dictionary with shell nutrient data
    """
    n_shells = len(shell_radii) - 1
    
    results = {
        'shell_inner': shell_radii[:-1],
        'shell_outer': shell_radii[1:],
        'shell_mid': (shell_radii[:-1] + shell_radii[1:]) / 2,
        'nutrient_before_avg': [],
        'nutrient_after_avg': [],
        'nutrient_delta_avg': [],
    }
    
    # Get voxel center positions
    Xc, Yc, Zc = grid.rebuild_center_mesh()
    
    # Calculate distance of each voxel from center
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    distances = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # For each shell, calculate average nutrient concentrations
    for i in range(n_shells):
        r_inner = float(shell_radii[i])
        r_outer = float(shell_radii[i + 1])
        
        # Mask for voxels in this shell
        in_shell = (distances >= r_inner) & (distances < r_outer)
        # Also exclude solid and hypha voxels
        liquid_in_shell = in_shell & (~grid.solid_mask) & (~grid.hypha_mask)
        
        # Calculate averages for before and after
        if np.any(liquid_in_shell):
            before_vals = nutrient_before[liquid_in_shell]
            after_vals = nutrient_after[liquid_in_shell]
            
            # Filter out NaN values
            valid_before = np.isfinite(before_vals)
            valid_after = np.isfinite(after_vals)
            valid = valid_before & valid_after
            
            if np.any(valid):
                avg_before = float(np.mean(before_vals[valid]))
                avg_after = float(np.mean(after_vals[valid]))
                avg_delta = avg_before - avg_after
            else:
                avg_before = 0.0
                avg_after = 0.0
                avg_delta = 0.0
        else:
            avg_before = 0.0
            avg_after = 0.0
            avg_delta = 0.0
        
        results['nutrient_before_avg'].append(avg_before)
        results['nutrient_after_avg'].append(avg_after)
        results['nutrient_delta_avg'].append(avg_delta)
    
    return results


def plot_shell_cross_section(center_um, graphs, shell_data, highlight_shell_idx=None):
    """
    Plot cross-section through center showing shells and hyphae.
    If highlight_shell_idx is provided, highlight edges in that shell in red.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw shells as circles (cross-section)
    shell_radii = shell_data['shell_radii']
    for r in shell_radii:
        circle = plt.Circle((center_um[0], center_um[1]), r, fill=False, 
                           edgecolor='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.add_patch(circle)
    
    # If highlighting a specific shell, draw it thicker
    if highlight_shell_idx is not None:
        r_inner = shell_data['shell_inner'][highlight_shell_idx]
        r_outer = shell_data['shell_outer'][highlight_shell_idx]
        for r in [r_inner, r_outer]:
            circle = plt.Circle((center_um[0], center_um[1]), r, fill=False, 
                               edgecolor='blue', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
    
    # Draw all hyphae in gray/black
    for cid, G in graphs.items():
        for u, v in G.edges:
            pu = node_pos(G, u)
            pv = node_pos(G, v)
            # Project to XY plane for cross-section
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], 'k-', linewidth=0.5, alpha=0.3)
    
    # If highlighting, draw edges in that shell in red
    if highlight_shell_idx is not None:
        edges_in_shell = shell_data['edges_per_shell'][highlight_shell_idx]
        for cid, u, v, pu, pv, length in edges_in_shell:
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], 'r-', linewidth=1.5, alpha=0.8)
    
    # Mark center
    ax.plot(center_um[0], center_um[1], 'k*', markersize=15, label='Pellet center')
    
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    if highlight_shell_idx is not None:
        ax.set_title(f'Cross-section with shell {highlight_shell_idx} highlighted (red = edges in shell)')
    else:
        ax.set_title('Cross-section showing all shells')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_hyphal_density_analysis(shell_data_list, timesteps_list):
    """
    Plot hyphal fraction vs distance from center for multiple timesteps.
    
    Args:
        shell_data_list: List of shell_data dictionaries, one per timestep
        timesteps_list: List of timestep indices corresponding to each shell_data
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a colormap for different timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, len(shell_data_list)))
    
    half_spawn_box = CONIDIA_SPAWN_BOX_EXTENT_UM / 2 * np.sqrt(3)
    
    for i, (shell_data, step) in enumerate(zip(shell_data_list, timesteps_list)):
        distances = np.array(shell_data['shell_mid'])
        fractions = np.array(shell_data['hyphal_fractions'])
        time_hr = step * TIME_STEP_HR
        
        # Find points within half of CONIDIA_SPAWN_BOX_EXTENT_UM and calculate average
        mask_avg = distances <= half_spawn_box
        
        if np.any(mask_avg):
            avg_fraction = np.mean(fractions[mask_avg])
            # Draw horizontal line from 0 to half_spawn_box
            ax.plot([0, half_spawn_box], [avg_fraction, avg_fraction],
                    '-', linewidth=2, markersize=6, color=colors[i])
        else:
            avg_fraction = None
        
        # Plot only points beyond half_spawn_box
        mask_plot = distances > half_spawn_box
        if np.any(mask_plot):
            distances_to_plot = distances[mask_plot]
            fractions_to_plot = fractions[mask_plot]
            
            # If we have an average, connect it to the first plotted point
            if avg_fraction is not None:
                # Prepend the endpoint of the average line to create smooth connection
                distances_to_plot = np.concatenate([[half_spawn_box], distances_to_plot])
                fractions_to_plot = np.concatenate([[avg_fraction], fractions_to_plot])
            
            ax.plot(distances_to_plot, fractions_to_plot, 
                    'o-', linewidth=2, markersize=6, 
                    color=colors[i], label=f'Step {step} ({time_hr:.1f} hr)')
        elif avg_fraction is not None:
            # Only average line exists, add label
            ax.plot([], [], 'o-', linewidth=2, markersize=6, 
                    color=colors[i], label=f'Step {step} ({time_hr:.1f} hr)')
    
    ax.set_xlabel('Distance from center (µm)', fontsize=12)
    ax.set_ylabel('Hyphal fraction (dimensionless)', fontsize=12)
    ax.set_title('Hyphal Fraction vs Distance from Center (Multiple Timesteps)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_nutrient_depletion_analysis(nutrient_data_list, timesteps_list, nutrient_name):
    """
    Plot nutrient depletion (delta) vs distance from center for multiple timesteps.
    
    Args:
        nutrient_data_list: List of nutrient data dictionaries, one per timestep
        timesteps_list: List of timestep indices
        nutrient_name: Name of the nutrient being plotted
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a colormap for different timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, len(nutrient_data_list)))
    
    half_spawn_box = CONIDIA_SPAWN_BOX_EXTENT_UM / 2 * np.sqrt(3)
    
    for i, (nutrient_data, step) in enumerate(zip(nutrient_data_list, timesteps_list)):
        distances = np.array(nutrient_data['shell_mid'])
        deltas = np.array(nutrient_data['nutrient_delta_avg'])
        time_hr = step * TIME_STEP_HR
        
        # Find points within half_spawn_box and calculate average
        mask_avg = distances <= half_spawn_box
        
        if np.any(mask_avg):
            avg_delta = np.mean(deltas[mask_avg])
            # Draw horizontal line from 0 to half_spawn_box
            ax.plot([0, half_spawn_box], [avg_delta, avg_delta],
                    '-', linewidth=2, markersize=6, color=colors[i])
        else:
            avg_delta = None
        
        # Plot only points beyond half_spawn_box
        mask_plot = distances > half_spawn_box
        if np.any(mask_plot):
            distances_to_plot = distances[mask_plot]
            deltas_to_plot = deltas[mask_plot]
            
            # If we have an average, connect it to the first plotted point
            if avg_delta is not None:
                # Prepend the endpoint of the average line to create smooth connection
                distances_to_plot = np.concatenate([[half_spawn_box], distances_to_plot])
                deltas_to_plot = np.concatenate([[avg_delta], deltas_to_plot])
            
            ax.plot(distances_to_plot, deltas_to_plot, 
                    'o-', linewidth=2, markersize=6, 
                    color=colors[i], label=f'Step {step} ({time_hr:.1f} hr)')
        elif avg_delta is not None:
            # Only average line exists, add label
            ax.plot([], [], 'o-', linewidth=2, markersize=6, 
                    color=colors[i], label=f'Step {step} ({time_hr:.1f} hr)')
    
    ax.set_xlabel('Distance from center (µm)', fontsize=12)
    ax.set_ylabel(f'{nutrient_name} depletion (g/L)', fontsize=12)
    ax.set_title(f'{nutrient_name} Depletion vs Distance from Center (Multiple Timesteps)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def analyze_citric_acid_by_shell(center_um, grid, citric_spatial_before, citric_spatial_after, shell_radii):
    """
    Analyze citric acid production by shell.
    
    Args:
        center_um: Center point for shells
        grid: VoxelGrid3D object
        citric_spatial_before: 3D array of cumulative citric acid (mg) before this timestep
        citric_spatial_after: 3D array of cumulative citric acid (mg) after this timestep
        shell_radii: Array of shell boundary radii
    
    Returns:
        Dictionary with shell citric acid data
    """
    n_shells = len(shell_radii) - 1
    
    results = {
        'shell_inner': shell_radii[:-1],
        'shell_outer': shell_radii[1:],
        'shell_mid': (shell_radii[:-1] + shell_radii[1:]) / 2,
        'citric_production_mg': [],
    }
    
    # Get voxel center positions
    Xc, Yc, Zc = grid.rebuild_center_mesh()
    
    # Calculate distance of each voxel from center
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    distances = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Calculate citric acid production delta
    citric_delta = citric_spatial_after - citric_spatial_before
    
    # For each shell, sum citric acid production
    for i in range(n_shells):
        r_inner = float(shell_radii[i])
        r_outer = float(shell_radii[i + 1])
        
        # Mask for voxels in this shell
        in_shell = (distances >= r_inner) & (distances < r_outer)
        
        # Sum citric acid production in this shell
        if np.any(in_shell):
            total_citric_mg = float(np.sum(citric_delta[in_shell]))
        else:
            total_citric_mg = 0.0
        
        results['citric_production_mg'].append(total_citric_mg)
    
    return results


def plot_citric_acid_production_analysis(citric_data_list, timesteps_list):
    """
    Plot citric acid production vs distance from center for multiple timesteps.
    No averaging - just plot all points.
    
    Args:
        citric_data_list: List of citric acid data dictionaries, one per timestep
        timesteps_list: List of timestep indices
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a colormap for different timesteps
    colors = plt.cm.viridis(np.linspace(0, 1, len(citric_data_list)))
    
    for i, (citric_data, step) in enumerate(zip(citric_data_list, timesteps_list)):
        distances = np.array(citric_data['shell_mid'])
        production = np.array(citric_data['citric_production_mg'])
        time_hr = step * TIME_STEP_HR
        
        ax.plot(distances, production, 
                'o-', linewidth=2, markersize=6, 
                color=colors[i], label=f'Step {step} ({time_hr:.1f} hr)')
    
    ax.set_xlabel('Distance from center (µm)', fontsize=12)
    ax.set_ylabel('Citric acid production (mg)', fontsize=12)
    ax.set_title('Citric Acid Production vs Distance from Center (Multiple Timesteps)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


def export_shell_data_csv(shell_data):
    """Export shell analysis data to CSV"""
    rows = []
    for i in range(len(shell_data['shell_mid'])):
        rows.append({
            'shell_index': i,
            'r_inner_um': float(shell_data['shell_inner'][i]),
            'r_outer_um': float(shell_data['shell_outer'][i]),
            'r_mid_um': float(shell_data['shell_mid'][i]),
            'shell_volume_um3': float(shell_data['shell_volumes'][i]),
            'hyphal_length_um': float(shell_data['hyphal_lengths'][i]),
            'hyphal_volume_um3': float(shell_data['hyphal_volumes'][i]),
            'hyphal_fraction': float(shell_data['hyphal_fractions'][i]),
        })
    
    write_csv(
        'shell_density_analysis.csv',
        fieldnames=['shell_index', 'r_inner_um', 'r_outer_um', 'r_mid_um', 
                   'shell_volume_um3', 'hyphal_length_um', 'hyphal_volume_um3', 
                   'hyphal_fraction'],
        rows=rows
    )


# =============================================================================
# MAIN LOOP
# =============================================================================

def run_simulation():
    _ensure_outdir()
    rng = np.random.default_rng(RNG_SEED)

    grid = VoxelGrid3D(extent_um=EXTENT_UM, n=N_GRID)
    initialize_nutrients(grid)

    plot_axes_and_grid(grid)

    conidium_radius = CONIDIUM_DIAMETER_UM / 2.0
    conidia = place_conidia(
        grid,
        n_conidia=N_CONIDIA,
        radius_um=conidium_radius,
        min_sep_um=MIN_CONIDIA_SEPARATION_UM,
        seed=RNG_SEED + 1
    )
    for c in conidia:
        carve_sphere_from_liquid(grid, c, conidium_radius)

    hypha_starts = assign_starting_hyphae(conidia, conidium_radius, seed=RNG_SEED + 2)
    graphs, start_dirs = initialize_hypha_graphs(grid, hypha_starts)

    plot_spores_and_initial_dirs(conidia, hypha_starts, conidium_radius)

    history = {
        "total_length_um": [],
        "total_volume_um3": [],
        "total_branch_nodes": [],
        "total_tips": [],
        # NEW:
        "citric_total_mg": [],
        "citric_mg_per_L": [],
        "liquid_volume_L": [],
        "mean_nutrients": {k: [] for k in NUTRIENT_SPECS.keys()},
        "apical_branch_events": [],
        "lateral_branch_events": [],
    }

    branch_event_counters = {"apical": 0, "lateral": 0}
    
    # Store graph snapshots for density analysis at 5 timesteps
    snapshot_timesteps = [int(N_STEPS * i / 5) for i in range(1, 6)]  # 20%, 40%, 60%, 80%, 100%
    graph_snapshots = {}  # {timestep: deep_copy_of_graphs}
    nutrient_snapshots = {}  # {timestep: {nutrient_name: {'before': array, 'after': array}}}
    citric_snapshots = {}  # {timestep: {'before': array, 'after': array}}

    for step in range(N_STEPS):
        per_node_actions = {}  # (cid, node_id) -> "GROW" or "SECRETE"

        # --- TIP growth / apical branching ---
        for cid, G in graphs.items():
            if G.number_of_nodes() == 0:
                continue

            tips = get_growing_tips(G, root=0)
            rng.shuffle(tips)

            for tip_node in tips:
                tip = node_pos(G, tip_node)

                prev = None
                if G.number_of_nodes() == 1:
                    prev = None
                elif G.degree[tip_node] == 1:
                    prev_node = next(iter(G.neighbors(tip_node)))
                    prev = node_pos(G, prev_node)

                default_dir = start_dirs.get(cid, None)
                branch_clock = int(G.nodes[tip_node].get("branch_clock_step", 0))

                dec = check_local_conditions(
                    grid=grid,
                    cid=cid,
                    node_um=tip,
                    prev_um=prev,
                    first_node_default_dir_unit=default_dir,
                    step_idx=step,
                    branch_clock_step=branch_clock,
                    rng=rng
                )
                out, tag = decision(grid, G, cid, tip_node, dec, step, branch_event_counters)
                per_node_actions[(cid, tip_node)] = tag

        # --- LATERAL branching pass (body nodes) ---
        for cid, G in graphs.items():
            if G.number_of_nodes() == 0:
                continue
            default_dir = start_dirs.get(cid, random_unit_vector(rng))
            lateral_outcomes = lateral_branch_attempts(
                grid=grid,
                G=G,
                cid=cid,
                start_dir=default_dir,
                step_idx=step,
                rng=rng,
                branch_event_counters=branch_event_counters
            )
            for (body_node, out, tag) in lateral_outcomes:
                per_node_actions[(cid, body_node)] = tag

        # --- Save nutrient state before depletion if this is a snapshot timestep ---
        if step in snapshot_timesteps:
            import copy
            nutrient_snapshots[step] = {}
            for nutr in ['Nitrogen', 'Phosphorus', 'Glucose']:
                nutrient_snapshots[step][nutr] = {
                    'before': copy.deepcopy(grid.nutrients[nutr])
                }
            # Save citric acid spatial state before depletion
            citric_snapshots[step] = {
                'before': copy.deepcopy(grid.metabolites["citric_acid_spatial_mg"])
            }

        # --- Action-local depletion around active nodes ---
        # This is now ALSO where citric acid is produced from glucose consumption.
        apply_action_local_depletion(grid, graphs, per_node_actions, use_26_neighbors=True)

        # --- Save nutrient state after depletion if this is a snapshot timestep ---
        if step in snapshot_timesteps:
            for nutr in ['Nitrogen', 'Phosphorus', 'Glucose']:
                nutrient_snapshots[step][nutr]['after'] = copy.deepcopy(grid.nutrients[nutr])
            # Save citric acid spatial state after depletion
            citric_snapshots[step]['after'] = copy.deepcopy(grid.metabolites["citric_acid_spatial_mg"])

        # --- Record mean nutrients after depletion ---
        means = mean_nutrients_over_liquid(grid)
        for nutr in NUTRIENT_SPECS.keys():
            history["mean_nutrients"][nutr].append(means.get(nutr, float("nan")))

        # --- Record metrics ---
        m = compute_metrics(grid, graphs)
        history["total_length_um"].append(m["total_length_um"])
        history["total_volume_um3"].append(m["total_volume_um3"])
        history["total_branch_nodes"].append(m["total_branch_nodes"])
        history["total_tips"].append(m["total_tips"])

        history["citric_total_mg"].append(m["citric_total_mg"])
        history["citric_mg_per_L"].append(m["citric_mg_per_L"])
        history["liquid_volume_L"].append(m["liquid_volume_L"])

        history["apical_branch_events"].append(branch_event_counters["apical"])
        history["lateral_branch_events"].append(branch_event_counters["lateral"])

        diffuse_nutrients(grid, seed=RNG_SEED + 1000 + step)
        
        # Save graph snapshot if this is one of the selected timesteps
        if step in snapshot_timesteps:
            graph_snapshots[step] = copy.deepcopy(graphs)
            print(f"  [Saved graph and nutrient snapshots at step {step}]")

        print(
            f"step={step} tips={m['total_tips']} branches(deg>=3)={m['total_branch_nodes']} "
            f"apical={branch_event_counters['apical']} lateral={branch_event_counters['lateral']} "
            f"citric={m['citric_mg_per_L']:.6g} mg/L"
        )

    return grid, graphs, history, graph_snapshots, nutrient_snapshots, citric_snapshots


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    grid, graphs, history, graph_snapshots, nutrient_snapshots, citric_snapshots = run_simulation()

    plot_skeletons_3d_with_node_age(graphs, grid)
    plot_time_series(history)
    
    # Hyphal density analysis for multiple timesteps
    print("\n=== Performing hyphal density analysis ===")
    pellet_center = grid.origin_um  # Center of conidia spawn box
    
    # Analyze density for each saved timestep
    shell_data_list = []
    timesteps_list = sorted(graph_snapshots.keys())
    
    # Get shell radii from final timestep
    max_radius, shell_radii = compute_shell_boundaries(pellet_center, graphs, N_SHELLS)
    
    for step in timesteps_list:
        print(f"Analyzing hyphal density at step {step}...")
        shell_data = analyze_hyphal_density_by_shell(pellet_center, graph_snapshots[step], n_shells=N_SHELLS)
        shell_data_list.append(shell_data)
    
    # Export shell data for final timestep
    export_shell_data_csv(shell_data_list[-1])
    
    # Cross-section showing all shells (final timestep)
    print("Plotting cross-section with all shells (final timestep)...")
    plot_shell_cross_section(pellet_center, graphs, shell_data_list[-1], highlight_shell_idx=None)
    
    # Cross-section highlighting a middle shell (final timestep)
    middle_shell_idx = len(shell_data_list[-1]['shell_mid']) // 2
    print(f"Plotting cross-section with shell {middle_shell_idx} highlighted (final timestep)...")
    plot_shell_cross_section(pellet_center, graphs, shell_data_list[-1], highlight_shell_idx=middle_shell_idx)
    
    # Hyphal fraction plot for all timesteps
    print("Plotting hyphal fraction for multiple timesteps...")
    plot_hyphal_density_analysis(shell_data_list, timesteps_list)
    
    # Nutrient depletion analysis
    print("\n=== Performing nutrient depletion analysis ===")
    for nutrient_name in ['Nitrogen', 'Phosphorus', 'Glucose']:
        print(f"Analyzing {nutrient_name} depletion...")
        nutrient_data_list = []
        
        for step in timesteps_list:
            nutrient_before = nutrient_snapshots[step][nutrient_name]['before']
            nutrient_after = nutrient_snapshots[step][nutrient_name]['after']
            
            nutrient_data = analyze_nutrient_by_shell(
                pellet_center, grid, nutrient_before, nutrient_after, 
                nutrient_name, shell_radii
            )
            nutrient_data_list.append(nutrient_data)
        
        print(f"Plotting {nutrient_name} depletion for multiple timesteps...")
        plot_nutrient_depletion_analysis(nutrient_data_list, timesteps_list, nutrient_name)
    
    # Citric acid production analysis
    print("\n=== Performing citric acid production analysis ===")
    citric_data_list = []
    
    for step in timesteps_list:
        print(f"Analyzing citric acid production at step {step}...")
        citric_before = citric_snapshots[step]['before']
        citric_after = citric_snapshots[step]['after']
        
        citric_data = analyze_citric_acid_by_shell(
            pellet_center, grid, citric_before, citric_after, shell_radii
        )
        citric_data_list.append(citric_data)
    
    print("Plotting citric acid production for multiple timesteps...")
    plot_citric_acid_production_analysis(citric_data_list, timesteps_list)
        
    print(f"Plotting {nutrient_name} depletion for multiple timesteps...")
    plot_nutrient_depletion_analysis(nutrient_data_list, timesteps_list, nutrient_name)
