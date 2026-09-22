import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# GLOBAL TUNABLES
# =============================================================================

# ---- Output ----
OUTPUT_DIR = "sim_outputs"

# ---- Grid / domain (µm) ----
EXTENT_UM = 100.0
CONIDIA_SPAWN_BOX_EXTENT_UM = 60.0
N_GRID = 90
RNG_SEED = 8

# Grid visualization (drawing every grid line at N=70 is too heavy)
GRID_PLOT_STRIDE = 5

# ---- Simulation ----
N_STEPS = 10
TIME_STEP_HR = 1.0   # x-axis on time-series plots

# ---- Nutrients (mM) ----
NUTRIENT_SPECS = {
    "Nitrogen":   {"mean": 37.8,  "pm": 1.89},   
    "Phosphorus": {"mean": 1.10,   "pm": 0.055},    
    "Glucose":    {"mean": 777.10,  "pm": 38.855},     
    "Oxygen":     {"mean": 0.2109,  "pm": 0.010545},  
}

SIGMA_SCALE = {
    "Nitrogen":   0.33,
    "Phosphorus": 0.33,
    "Glucose":    0.33,
    "Oxygen":     0.33,
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
INITIAL_DIR_CONE_DEG = 90.0   # initial growth direction must lie within this cone of the inward (toward-origin) direction

# ---- Hypha growth ----
MU_MAX = 9.99            # maximum tip growth rate (µm/hr)
DEPLETION_RADIUS = 5.0  # radius (µm) around a tip used for sensing & depletion
HYPHA_RADIUS_UM = 1.3   # physical radius of the hypha cylinder (µm)

# ---- Apical branching ----
PSI_BRANCH = 0.96111             # branch when  ψ · μ_{t-1} ≥ μ_t  (drop-in-rate threshold)
MIN_BRANCH_ANGLE_DEG = 50.0   # minimum angle (deg) between the two daughter directions
APICAL_BRANCH_MIN_STEP = 3    # apical branching disabled while step_idx < this

# ---- Lateral branching ----
LATERAL_DISTANCE_FROM_TIP = 4.6   # min distance (µm) from any tip for a lateral branching site
PHI_N = 0.15                      # local N ≥ PHI_N · [N]_init required for a lateral branch
PHI_P = 0.15                      # local P ≥ PHI_P · [P]_init required for a lateral branch
LATERAL_BRANCH_MIN_STEP = 7       # lateral branching disabled while step_idx < this

# ---- Trajectory bias ----
MAX_DEVIATION_DEG = 75.0     # new growth direction can deviate ≤ this from the tip's current direction

# ---- Nutrient diffusion (∂C/∂t = D ∇²C  − q) ----
# Real diffusion coefficients in water at room temperature (m²/s). Used as
# RATIOS to set the relative diffusion speed between species. The absolute
# per-step strength is controlled by DIFFUSION_STRENGTH — see diffuse_nutrients.
DIFFUSION_COEFFS_M2_PER_S = {
    "Nitrogen":   2.61e-9,
    "Phosphorus": 6.05e-10,    # phosphate
    "Glucose":    6.0e-10,
    "Oxygen":     2.12e-9,
}
DIFFUSION_STRENGTH = 0.1       # α for the fastest species (must be ≤ 1/6 for stability in 3D)

# ---- Biomass / consumption geometry ----
MYCELIUM_DENSITY_KG_PER_M3 = 150.0   # dry-weight density of mycelium
SUBAPICAL_UPTAKE_FACTOR = 0.1        # subapical (non-tip) uptake = factor × tip MM rate

# ---- Citric acid production ----
EPSILON_N = 0.378                 # local mean N (mM) ≤ this triggers citric acid production (was 0.05 g/L)
EPSILON_P = 0.0735                # local mean P (mM) ≤ this triggers citric acid production (was 0.01 g/L)
CITRIC_mmol_per_mol_GLUCOSE = 702.34   # mmol citric acid produced per mol glucose consumed (λ_CA, was 749 mg/g)

# ---- Nutrient uptake (Michaelis–Menten kinetics) ----
# Highest uptake rates v_max in mmol·gDW⁻¹·h⁻¹; Michaelis constants K_m in mM.
# Used by the nutrient depletion / consumption rule (to be wired in next).

# Phosphorus
V_MAX_P = 0.08          # highest uptake rate of phosphorus (mmol·gDW⁻¹·h⁻¹)
K_M_P   = 0.0333        # Michaelis constant of phosphorus (mM)

# Nitrogen
V_MAX_N = 0.2922        # highest uptake rate of nitrogen (mmol·gDW⁻¹·h⁻¹)
K_M_N   = 0.0735        # Michaelis constant of nitrogen (mM)

# Oxygen
V_MAX_O = 12825.0       # highest uptake rate of oxygen (mmol·gDW⁻¹·h⁻¹)
K_M_O   = 0.0156        # Michaelis constant of oxygen (mM)

# Glucose — dual-pathway transport
V_G1     = 0.00031419   # glucose passive uptake rate (mmol·gDW⁻¹·h⁻¹)
V_G2_MAX = 0.186        # max rate of glucose high-affinity transport (mmol·gDW⁻¹·h⁻¹)
K_G2     = 0.26         # Michaelis constant of glucose high-affinity transport (mM)
K_I2     = 933.0        # citrate inhibition constant of glucose high-affinity transport (mM)


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
            # Per-conidium cumulative citric acid produced (mmol; sized in initialize_hypha_graphs)
            "citric_acid_total_mmol": None,
            # Per-voxel citric acid concentration (mM); read by the glucose dual-pathway
            # equation as [C]_local for the citrate-inhibition term.
            "citric_acid_spatial_mM": np.zeros((self.n, self.n, self.n), dtype=np.float64),
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


def _inward_biased_unit_vector(rng, inward_dir, max_angle_deg, max_tries=200):
    """Random unit vector within `max_angle_deg` of `inward_dir` (rejection sampling)."""
    inward = np.asarray(inward_dir, dtype=float)
    n_inward = float(np.linalg.norm(inward))
    if n_inward < 1e-12:
        return random_unit_vector(rng)
    inward = inward / n_inward
    cos_max = float(np.cos(np.deg2rad(max_angle_deg)))
    for _ in range(int(max_tries)):
        v = random_unit_vector(rng)
        if float(np.dot(v, inward)) >= cos_max:
            return v
    return inward


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

def assign_starting_hyphae(conidia_centers_um, conidium_radius_um, seed,
                           origin_um=(0.0, 0.0, 0.0)):
    rng = np.random.default_rng(seed)
    n = len(conidia_centers_um)
    p_miss = MISS_HYPHA_BASE_PROB + MISS_HYPHA_PER_CONIDIUM * max(0, n - 1)
    p_miss = float(np.clip(p_miss, 0.0, MISS_HYPHA_MAX_PROB))

    origin = np.asarray(origin_um, dtype=float)

    starts = []
    for cid, center in enumerate(conidia_centers_um):
        if rng.random() < p_miss:
            starts.append({"conidium_id": cid, "has_hypha": False,
                           "start_point_um": None, "direction_unit": None})
            continue
        inward = origin - np.asarray(center, dtype=float)
        normal = _inward_biased_unit_vector(rng, inward, INITIAL_DIR_CONE_DEG)
        start_point = center + conidium_radius_um * normal
        starts.append({"conidium_id": cid, "has_hypha": True,
                       "start_point_um": start_point, "direction_unit": normal})
    return starts


# =============================================================================
# SKELETON GRAPHS
# =============================================================================

def initialize_hypha_graphs(grid: VoxelGrid3D, hypha_starts):
    n_conidia = len(hypha_starts)
    grid.metabolites["citric_acid_total_mmol"] = np.zeros(n_conidia, dtype=np.float64)

    graphs = {cid: nx.Graph() for cid in range(n_conidia)}
    start_dirs = {}

    for item in hypha_starts:
        cid = item["conidium_id"]
        if not item["has_hypha"]:
            continue

        p0 = np.array(item["start_point_um"], dtype=float)
        d0 = np.array(item["direction_unit"], dtype=float)
        graphs[cid].add_node(
            0,
            pos=p0,
            birth_step=0,
            branch_clock_step=0,
            branch_origin="root",
            last_dir=d0,
        )
        start_dirs[cid] = d0

        idx = grid.world_to_index(*p0)
        if idx is not None:
            i, j, k = idx
            grid.hypha_mask[i, j, k] = True
            grid.hypha_owner[i, j, k] = cid

    return graphs, start_dirs


def node_pos(G: nx.Graph, n) -> np.ndarray:
    return np.array(G.nodes[n]["pos"], dtype=float)


def get_growing_tips(G: nx.Graph, root: int = 0) -> list:
    if G.number_of_nodes() == 0:
        return []
    if G.number_of_nodes() == 1:
        return [next(iter(G.nodes))]
    return [n for n in G.nodes if n != root and G.degree[n] == 1]


# =============================================================================
# RULE OF GROWTH: TIP ELONGATION
# =============================================================================
#
#                μ_max  / N_cur     P_cur     O_cur \
#   μ(tip)  =  ─────── ( ─────  +  ─────  +  ───── )
#                  3    \ N_init   P_init    O_init /
#
# - N_cur, P_cur, O_cur = mean concentrations across LIQUID voxels (not solid,
#   not hypha) inside a sphere of radius DEPLETION_RADIUS around the tip.
# - N_init, P_init, O_init = NUTRIENT_SPECS[*]["mean"].
# - dt = 1 hr per step, so step length (µm) = μ.
# - Direction: toward the LIQUID voxel inside the same sphere whose normalized
#   sum (N/N_init + P/P_init + O/O_init) is largest.

def _voxels_within_sphere(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float) -> np.ndarray:
    Xc, Yc, Zc = grid.rebuild_center_mesh()
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    return (dx * dx + dy * dy + dz * dz) <= (radius_um * radius_um)


def _liquid_sphere_mask(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float) -> np.ndarray:
    sphere = _voxels_within_sphere(grid, center_um, radius_um)
    return sphere & (~grid.solid_mask) & (~grid.hypha_mask)


def mean_local_in_sphere(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float, names) -> dict:
    """Mean of named nutrients over LIQUID voxels in a sphere around `center_um`."""
    liquid = _liquid_sphere_mask(grid, center_um, radius_um)
    means = {}
    for name in names:
        arr = grid.nutrients[name]
        valid = liquid & np.isfinite(arr)
        means[name] = float(np.mean(arr[valid])) if np.any(valid) else 0.0
    return means


def mean_local_NPO(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float) -> dict:
    """Mean of N, P, O over LIQUID voxels in a sphere around `center_um`."""
    return mean_local_in_sphere(grid, center_um, radius_um,
                                ("Nitrogen", "Phosphorus", "Oxygen"))


def mean_local_citric(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float) -> float:
    """Mean citric acid concentration (mM) over LIQUID voxels in a sphere around `center_um`."""
    liquid = _liquid_sphere_mask(grid, center_um, radius_um)
    if not np.any(liquid):
        return 0.0
    arr = grid.metabolites["citric_acid_spatial_mM"]
    return float(np.mean(arr[liquid]))


def growth_rate_from_local(means: dict) -> float:
    """μ = (μ_max / 3) · (N_cur/N_init + P_cur/P_init + O_cur/O_init)."""
    N_init = float(NUTRIENT_SPECS["Nitrogen"]["mean"])
    P_init = float(NUTRIENT_SPECS["Phosphorus"]["mean"])
    O_init = float(NUTRIENT_SPECS["Oxygen"]["mean"])
    ratio_sum = (
        means["Nitrogen"]   / max(N_init, 1e-12) +
        means["Phosphorus"] / max(P_init, 1e-12) +
        means["Oxygen"]     / max(O_init, 1e-12)
    )
    return (MU_MAX / 3.0) * ratio_sum


def best_target_voxel(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float):
    """Index of the LIQUID voxel inside the sphere with the highest normalized N+P+O."""
    liquid = _liquid_sphere_mask(grid, center_um, radius_um)
    cand_idx = np.argwhere(liquid)
    if cand_idx.size == 0:
        return None

    N_init = float(NUTRIENT_SPECS["Nitrogen"]["mean"])
    P_init = float(NUTRIENT_SPECS["Phosphorus"]["mean"])
    O_init = float(NUTRIENT_SPECS["Oxygen"]["mean"])

    Ns = grid.nutrients["Nitrogen"][cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]
    Ps = grid.nutrients["Phosphorus"][cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]
    Os = grid.nutrients["Oxygen"][cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]

    finite = np.isfinite(Ns) & np.isfinite(Ps) & np.isfinite(Os)
    if not np.any(finite):
        return None
    cand_idx = cand_idx[finite]
    Ns, Ps, Os = Ns[finite], Ps[finite], Os[finite]

    score = (Ns / max(N_init, 1e-12)
             + Ps / max(P_init, 1e-12)
             + Os / max(O_init, 1e-12))
    best = int(np.argmax(score))
    return tuple(int(x) for x in cand_idx[best])


def top_k_score_voxels(grid: VoxelGrid3D, center_um: np.ndarray, radius_um: float, k=None):
    """Return LIQUID voxel indices ranked by descending normalized N+P+O score.

    If `k` is None, all candidate voxels are returned in score order.
    """
    liquid = _liquid_sphere_mask(grid, center_um, radius_um)
    cand_idx = np.argwhere(liquid)
    if cand_idx.size == 0:
        return []

    N_init = float(NUTRIENT_SPECS["Nitrogen"]["mean"])
    P_init = float(NUTRIENT_SPECS["Phosphorus"]["mean"])
    O_init = float(NUTRIENT_SPECS["Oxygen"]["mean"])

    Ns = grid.nutrients["Nitrogen"][cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]
    Ps = grid.nutrients["Phosphorus"][cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]
    Os = grid.nutrients["Oxygen"][cand_idx[:, 0], cand_idx[:, 1], cand_idx[:, 2]]

    finite = np.isfinite(Ns) & np.isfinite(Ps) & np.isfinite(Os)
    if not np.any(finite):
        return []
    cand_idx = cand_idx[finite]
    Ns, Ps, Os = Ns[finite], Ps[finite], Os[finite]

    score = (Ns / max(N_init, 1e-12)
             + Ps / max(P_init, 1e-12)
             + Os / max(O_init, 1e-12))

    total = int(score.shape[0])
    n = total if k is None else min(int(k), total)
    if n <= 0:
        return []
    top = np.argpartition(-score, n - 1)[:n]
    top = top[np.argsort(-score[top])]
    return [tuple(int(x) for x in cand_idx[i]) for i in top]


def _voxel_center_um(grid: VoxelGrid3D, ijk) -> np.ndarray:
    return np.array([
        grid.x_centers[ijk[0]],
        grid.y_centers[ijk[1]],
        grid.z_centers[ijk[2]],
    ], dtype=float)


def _unit_dir_from_to(tip_um: np.ndarray, target_um: np.ndarray):
    v = target_um - tip_um
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def _candidates_in_trajectory_cone(grid: VoxelGrid3D, tip_um: np.ndarray,
                                   prev_dir, candidates: list):
    """Return [(ijk, unit_dir)] for each candidate within the MAX_DEVIATION_DEG
    cone around `prev_dir`. Order is preserved (i.e., best score first).
    If `prev_dir` is None or zero, no cone constraint is applied.
    """
    cos_threshold = float(np.cos(np.deg2rad(MAX_DEVIATION_DEG)))

    prev_unit = None
    if prev_dir is not None:
        pv = np.asarray(prev_dir, dtype=float)
        pn = float(np.linalg.norm(pv))
        if pn > 1e-12:
            prev_unit = pv / pn

    out = []
    for cand in candidates:
        d = _unit_dir_from_to(tip_um, _voxel_center_um(grid, cand))
        if d is None:
            continue
        if prev_unit is not None and float(np.dot(prev_unit, d)) < cos_threshold:
            continue
        out.append((cand, d))
    return out


def grow_tip(grid: VoxelGrid3D, G: nx.Graph, cid: int, tip_node: int, step_idx: int, _means: dict = None):
    """Apply the tip-elongation rule once. Returns the new node id, or None if no growth.

    Direction is the highest-scoring liquid voxel in DEPLETION_RADIUS whose direction
    sits within MAX_DEVIATION_DEG of the tip's current trajectory (`last_dir`).
    """
    tip_um = node_pos(G, tip_node)

    means = _means if _means is not None else mean_local_NPO(grid, tip_um, DEPLETION_RADIUS)
    mu = growth_rate_from_local(means)
    if mu <= 0.0:
        return None

    prev_dir = G.nodes[tip_node].get("last_dir", None)
    candidates = top_k_score_voxels(grid, tip_um, DEPLETION_RADIUS, k=None)
    candidates_dirs = _candidates_in_trajectory_cone(grid, tip_um, prev_dir, candidates)
    if not candidates_dirs:
        return None

    direction = candidates_dirs[0][1]
    new_pos = tip_um + mu * direction  # dt = 1 hr → step length = μ µm

    idx = grid.world_to_index(*new_pos)
    if idx is None:
        return None
    i, j, k = idx
    if grid.solid_mask[i, j, k] or grid.hypha_mask[i, j, k]:
        return None

    new_id = (max(G.nodes) + 1) if G.number_of_nodes() > 0 else 0
    G.add_node(
        new_id,
        pos=new_pos,
        birth_step=step_idx,
        branch_origin="grow",
        last_mu=mu,
        last_dir=direction,
    )
    G.add_edge(tip_node, new_id)

    grid.hypha_mask[i, j, k] = True
    grid.hypha_owner[i, j, k] = cid
    return new_id


# =============================================================================
# RULE: NUTRIENT CONSUMPTION + CITRIC ACID PRODUCTION
# =============================================================================
#
# Per consumption source (tip node, or edge midpoint), all nutrients are taken
# up over a DEPLETION_RADIUS sphere of liquid voxels. The total consumed mass
# of each nutrient is distributed equally across those voxels and clamped ≥ 0.
#
# Biomass approximation (g DW):
#   tip         : cylinder of length=DEPLETION_RADIUS, radius=HYPHA_RADIUS_UM
#   subapical   : cylinder of length=edge_length,        radius=HYPHA_RADIUS_UM
#   rate_factor : 1.0 for tips, SUBAPICAL_UPTAKE_FACTOR (=0.1) for edges
#
# N, P, O — standard Michaelis–Menten:
#       v = v_max · [S] / (K_m + [S])      (mmol · gDW⁻¹ · h⁻¹)
#
# Glucose — dual-pathway (passive + high-affinity with citrate inhibition):
#       v_G = v_G1·[G]  +  v_G2,max·[G] / [(K_G2 + [G])·(1 + [C]/K_i2)]
#   - The passive term v_G1·[G] is ALWAYS active.
#   - The high-affinity term is only used at this consumption source if local
#     [N] ≤ ε_N OR [P] ≤ ε_P (the same condition that triggers CA production).
#
# Citric acid production:
#       mmol_CA = (active glucose mmol consumed) · CITRIC_mmol_per_mol_GLUCOSE / 1000
#   The CA mmol is added to the per-conidium total and distributed equally as
#   a Δ[C] (mM) across the same liquid sphere.

NUTRIENT_MM_PARAMS = {
    "Nitrogen":   (V_MAX_N, K_M_N),
    "Phosphorus": (V_MAX_P, K_M_P),
    "Oxygen":     (V_MAX_O, K_M_O),
}


def cylinder_biomass_g(length_um: float, radius_um: float) -> float:
    """Mass (g DW) of a hypha cylinder = π · r² · L · ρ_myc."""
    volume_m3 = float(np.pi) * (radius_um * 1e-6) ** 2 * (length_um * 1e-6)
    mass_kg = volume_m3 * MYCELIUM_DENSITY_KG_PER_M3
    return float(mass_kg * 1000.0)


def michaelis_menten(v_max: float, K_m: float, S: float) -> float:
    """v = v_max · [S] / (K_m + [S]). [S] clamped at 0."""
    S = max(0.0, float(S))
    denom = K_m + S
    if denom <= 0.0:
        return 0.0
    return float(v_max * S / denom)


def consume_at_point(grid: VoxelGrid3D, point_um: np.ndarray, biomass_g: float,
                     rate_factor: float = 1.0, cid: int = None) -> float:
    """Apply N/P/O/Glucose uptake at a consumption source.

    Returns mmol of citric acid produced (0.0 if the active glucose pathway
    is not engaged at this point).
    """
    liquid = _liquid_sphere_mask(grid, point_um, DEPLETION_RADIUS)
    cand_idx = np.argwhere(liquid)
    if cand_idx.size == 0:
        return 0.0

    n_cells = int(cand_idx.shape[0])
    voxel_vol_L = grid.voxel_volume_L()
    ii = cand_idx[:, 0]; jj = cand_idx[:, 1]; kk = cand_idx[:, 2]

    means = mean_local_in_sphere(
        grid, point_um, DEPLETION_RADIUS,
        ("Nitrogen", "Phosphorus", "Oxygen", "Glucose"),
    )

    # ---- N, P, O via standard Michaelis–Menten ----
    for name, (v_max, K_m) in NUTRIENT_MM_PARAMS.items():
        v = michaelis_menten(v_max, K_m, means[name])              # mmol·gDW⁻¹·h⁻¹
        consumed_mmol = v * biomass_g * float(rate_factor)          # × dt = 1 hr
        if consumed_mmol <= 0.0:
            continue
        delta_mM = (consumed_mmol / n_cells) / voxel_vol_L
        arr = grid.nutrients[name]
        arr[ii, jj, kk] = np.maximum(0.0, arr[ii, jj, kk] - delta_mM)
        grid.nutrients[name] = arr

    # ---- Glucose: dual-pathway (passive always; active only when CA fires) ----
    G = max(0.0, means["Glucose"])

    ca_active = (means["Nitrogen"] <= EPSILON_N) or (means["Phosphorus"] <= EPSILON_P)

    v_G_passive = V_G1 * G                                          # mmol·gDW⁻¹·h⁻¹
    v_G_active = 0.0
    if ca_active and G > 0.0:
        C = mean_local_citric(grid, point_um, DEPLETION_RADIUS)
        inhibition = 1.0 + (C / K_I2)
        denom = (K_G2 + G) * inhibition
        if denom > 0.0:
            v_G_active = V_G2_MAX * G / denom

    glucose_consumed_mmol = (v_G_passive + v_G_active) * biomass_g * float(rate_factor)
    if glucose_consumed_mmol > 0.0:
        delta_mM = (glucose_consumed_mmol / n_cells) / voxel_vol_L
        arr = grid.nutrients["Glucose"]
        arr[ii, jj, kk] = np.maximum(0.0, arr[ii, jj, kk] - delta_mM)
        grid.nutrients["Glucose"] = arr

    # ---- Citric acid produced from the ACTIVE glucose pathway only ----
    citric_mmol = 0.0
    if v_G_active > 0.0:
        active_glucose_mmol = v_G_active * biomass_g * float(rate_factor)
        # CITRIC_mmol_per_mol_GLUCOSE is mmol_CA per mol_G; divide by 1000 to scale to mmol_G.
        citric_mmol = active_glucose_mmol * CITRIC_mmol_per_mol_GLUCOSE / 1000.0

        if cid is not None:
            grid.metabolites["citric_acid_total_mmol"][cid] += citric_mmol

        delta_C_mM = (citric_mmol / n_cells) / voxel_vol_L
        grid.metabolites["citric_acid_spatial_mM"][ii, jj, kk] += delta_C_mM

    return citric_mmol


def consume_nutrients(grid: VoxelGrid3D, graphs: dict):
    """All-nutrient consumption pass. Returns (total_ca_mmol, n_ca_events)."""
    tip_biomass = cylinder_biomass_g(DEPLETION_RADIUS, HYPHA_RADIUS_UM)

    total_ca_mmol = 0.0
    n_ca_events = 0

    # Tips at full rate
    for cid, G in graphs.items():
        if G.number_of_nodes() == 0:
            continue
        for tip in get_growing_tips(G, root=0):
            ca = consume_at_point(grid, node_pos(G, tip), tip_biomass,
                                  rate_factor=1.0, cid=cid)
            total_ca_mmol += ca
            if ca > 0.0:
                n_ca_events += 1

    # Subapical edges at SUBAPICAL_UPTAKE_FACTOR rate
    for cid, G in graphs.items():
        for u, v in G.edges:
            pu = node_pos(G, u)
            pv = node_pos(G, v)
            edge_length = float(np.linalg.norm(pv - pu))
            if edge_length < 1e-12:
                continue
            edge_midpoint = 0.5 * (pu + pv)
            edge_biomass = cylinder_biomass_g(edge_length, HYPHA_RADIUS_UM)
            ca = consume_at_point(grid, edge_midpoint, edge_biomass,
                                  rate_factor=SUBAPICAL_UPTAKE_FACTOR, cid=cid)
            total_ca_mmol += ca
            if ca > 0.0:
                n_ca_events += 1

    return total_ca_mmol, n_ca_events


# =============================================================================
# NUTRIENT DIFFUSION
# =============================================================================
#
# Reaction-diffusion PDE:  ∂C_i/∂t = ∇·(D_i ∇C_i)  −  q_i(C_i, X)
# The reaction term q_i is handled by `consume_nutrients`; here we apply the
# spatial diffusion operator with one explicit-Euler step per simulation step.
#
# Physics note
# ------------
# For dx ≈ 1.43 µm and the real D values, ONE 1-hour step would equilibrate
# the entire 100 µm domain (diffusion length √(D·t) is millimetres). To preserve
# the spatial gradients the model relies on, we use the real D values only to
# set RELATIVE diffusion speed between species, and scale absolute speed by a
# tunable knob:
#
#       α_i  =  DIFFUSION_STRENGTH · D_i / D_max
#       C_new  =  C_old + α_i · (Σ valid neighbours − n_valid · C_old)
#
# This is a discrete Laplacian update with no-flux boundaries (out-of-domain
# neighbours contribute 0 to both Σ and n_valid, equivalent to a ghost-cell
# treatment with C_ghost = 0, which is a leaky boundary; for a strict no-flux
# wall we'd want C_ghost = C_boundary, but the leak only matters at the cube
# faces and for our scenario doesn't meaningfully affect interior dynamics).
#
# Hypha voxels are reset to 0 after the update — the hypha is a perfect uptake
# sink. Solid voxels (conidium interiors) stay NaN.
# Concentrations are clamped at 0.

def diffuse_nutrients(grid: VoxelGrid3D):
    """One explicit-Euler 6-neighbour diffusion step per nutrient species."""
    if not DIFFUSION_COEFFS_M2_PER_S:
        return
    D_max = max(DIFFUSION_COEFFS_M2_PER_S.values())
    liquid = (~grid.solid_mask) & (~grid.hypha_mask)

    for nutr, arr in grid.nutrients.items():
        if nutr not in DIFFUSION_COEFFS_M2_PER_S:
            continue

        D = float(DIFFUSION_COEFFS_M2_PER_S[nutr])
        alpha = DIFFUSION_STRENGTH * D / D_max

        valid = liquid & np.isfinite(arr)
        arr_m = np.where(valid, arr.astype(np.float64), 0.0)
        cnt = valid.astype(np.float64)

        # Pad with zeros (1-voxel border) → no-flux boundary at the cube faces
        arr_p = np.pad(arr_m, 1, mode="constant", constant_values=0.0)
        cnt_p = np.pad(cnt,   1, mode="constant", constant_values=0.0)

        nbr_sum = (arr_p[2:, 1:-1, 1:-1] + arr_p[:-2, 1:-1, 1:-1] +
                   arr_p[1:-1, 2:, 1:-1] + arr_p[1:-1, :-2, 1:-1] +
                   arr_p[1:-1, 1:-1, 2:] + arr_p[1:-1, 1:-1, :-2])
        nbr_cnt = (cnt_p[2:, 1:-1, 1:-1] + cnt_p[:-2, 1:-1, 1:-1] +
                   cnt_p[1:-1, 2:, 1:-1] + cnt_p[1:-1, :-2, 1:-1] +
                   cnt_p[1:-1, 1:-1, 2:] + cnt_p[1:-1, 1:-1, :-2])

        # Discrete Laplacian-style update: C += α · (Σ valid neighbours − n_valid · C)
        delta = alpha * (nbr_sum - nbr_cnt * arr_m)

        new_arr = arr.astype(np.float32, copy=True)
        updated = np.maximum(0.0, arr_m + delta).astype(np.float32)
        new_arr[valid] = updated[valid]

        # Re-impose masks: hypha as 0 (consumption sink), solid as NaN
        new_arr[grid.hypha_mask] = 0.0
        new_arr[grid.solid_mask] = np.nan
        grid.nutrients[nutr] = new_arr


# =============================================================================
# METRICS
# =============================================================================

def total_edge_length_um(G: nx.Graph) -> float:
    L = 0.0
    for u, v in G.edges:
        L += float(np.linalg.norm(node_pos(G, v) - node_pos(G, u)))
    return L


def count_branch_nodes(G: nx.Graph) -> int:
    return sum(1 for n in G.nodes if G.degree[n] >= 3)


def count_tips(G: nx.Graph) -> int:
    return len(get_growing_tips(G, root=0))


def mean_nutrients_over_liquid(grid: VoxelGrid3D) -> dict:
    liquid_mask = (~grid.solid_mask) & (~grid.hypha_mask)
    means = {}
    for nutr, arr in grid.nutrients.items():
        m = liquid_mask & np.isfinite(arr)
        means[nutr] = float(np.mean(arr[m])) if np.any(m) else float("nan")
    return means


def compute_metrics(grid: VoxelGrid3D, graphs: dict) -> dict:
    total_length = 0.0
    total_branches = 0
    total_tips = 0
    n_hyphae = 0

    for _cid, G in graphs.items():
        if G.number_of_nodes() == 0:
            continue
        n_hyphae += 1
        total_length += total_edge_length_um(G)
        total_branches += count_branch_nodes(G)
        total_tips += count_tips(G)

    total_volume = total_length * float(np.pi) * (HYPHA_RADIUS_UM ** 2)
    denom = max(n_hyphae, 1)
    avg_length = total_length / denom
    avg_volume = total_volume / denom
    avg_tips = total_tips / denom

    citric_total_mmol = float(np.sum(grid.metabolites["citric_acid_total_mmol"]))
    liquid_L = grid.liquid_volume_L()
    citric_mM = (citric_total_mmol / liquid_L) if liquid_L > 0 else float("nan")

    return {
        "total_length_um":      total_length,
        "total_volume_um3":     total_volume,
        "avg_length_um":        avg_length,
        "avg_volume_um3":       avg_volume,
        "avg_tips_per_hypha":   avg_tips,
        "n_hyphae":             n_hyphae,
        "total_branch_nodes":   total_branches,
        "total_tips":           total_tips,
        "citric_total_mmol":    citric_total_mmol,
        "citric_mM":            citric_mM,
        "liquid_volume_L":      liquid_L,
    }


# =============================================================================
# RULE: APICAL BRANCHING
# =============================================================================
#
# Per tip each step:
#   IF  ψ · μ_{t-1} ≥ μ_t   THEN  create two apical daughters.
#
#   - μ_{t-1} = `last_mu` attribute on the tip node (set when the tip was
#     created, by either grow_tip or apical_branch). Starting nodes have no
#     last_mu, so they cannot apical-branch on step 0 — they grow instead.
#   - μ_t = current growth rate at the tip's position (Eq. for μ).
#   - First daughter is placed at distance μ_t in the direction of the
#     top-scoring LIQUID voxel by the normalized N+P+O score (same scoring
#     as ordinary growth), within DEPLETION_RADIUS of the parent.
#   - Second daughter direction is the next-best voxel whose direction makes
#     an angle ≥ MIN_BRANCH_ANGLE_DEG with the first. If none qualifies,
#     only the first daughter is placed (and step_tip counts it as GROW).
#   - Each daughter inherits last_mu = μ_t.
#
# If both daughter positions fail validation (out of bounds / inside solid /
# inside an existing hypha voxel), step_tip falls back to ordinary growth.

def apical_branch(grid: VoxelGrid3D, G: nx.Graph, cid: int, tip_node: int,
                  step_idx: int, mu_t: float) -> list:
    """Create up to 2 apical daughter tips with a minimum angular separation
    AND respecting the trajectory bias (≤ MAX_DEVIATION_DEG from the tip's
    current direction).

    Returns the list of newly added node ids.
    """
    tip_um = node_pos(G, tip_node)
    candidates = top_k_score_voxels(grid, tip_um, DEPLETION_RADIUS, k=None)
    if not candidates:
        return []

    prev_dir = G.nodes[tip_node].get("last_dir", None)
    candidates_dirs = _candidates_in_trajectory_cone(grid, tip_um, prev_dir, candidates)
    if not candidates_dirs:
        return []

    # First daughter: top-scoring direction within the cone
    first_dir = candidates_dirs[0][1]

    # Second daughter: next-best within the cone AND ≥ MIN_BRANCH_ANGLE_DEG from first
    cos_branch = float(np.cos(np.deg2rad(MIN_BRANCH_ANGLE_DEG)))
    second_dir = None
    for _cand, d in candidates_dirs[1:]:
        if float(np.dot(first_dir, d)) <= cos_branch:  # angle ≥ MIN_BRANCH_ANGLE_DEG
            second_dir = d
            break

    new_ids = []
    for direction in (first_dir, second_dir):
        if direction is None:
            continue
        new_pos = tip_um + mu_t * direction
        idx = grid.world_to_index(*new_pos)
        if idx is None:
            continue
        i, j, k = idx
        if grid.solid_mask[i, j, k] or grid.hypha_mask[i, j, k]:
            continue

        new_id = (max(G.nodes) + 1) if G.number_of_nodes() > 0 else 0
        G.add_node(
            new_id,
            pos=new_pos,
            birth_step=step_idx,
            branch_origin="apical",
            last_mu=mu_t,
            last_dir=direction,
        )
        G.add_edge(tip_node, new_id)
        grid.hypha_mask[i, j, k] = True
        grid.hypha_owner[i, j, k] = cid
        new_ids.append(new_id)

    return new_ids


# =============================================================================
# RULE: LATERAL BRANCHING
# =============================================================================
#
# Per body node b (non-root, non-tip) in each conidium graph, every step:
#
#   IF  [N]_local ≥ φ_N · [N]_init                         (Eq. 10)
#   AND [P]_local ≥ φ_P · [P]_init                         (Eq. 11)
#   AND  d  ≤  ‖tip_pos − b_pos‖   for every tip           (Eq. 12)
#   THEN create a lateral branch at b
#
# - Local sense uses the same DEPLETION_RADIUS sphere we use elsewhere.
# - The new lateral tip is placed at distance μ_t (computed at b) along the
#   highest-scoring direction in the sphere. No trajectory bias is applied
#   (lateral branches are new directions, not continuations).
# - The body node `b` keeps its own last_dir; the new lateral tip stores its
#   emergence direction in last_dir so subsequent growth respects bias.
# - One branch per qualifying body node per step. The rule rate-limits itself
#   because a fresh lateral creates a new "tip" near b, so b's distance check
#   fails until the new chain grows ≥ d away.

def lateral_branching(grid: VoxelGrid3D, G: nx.Graph, cid: int, step_idx: int) -> list:
    """Spawn lateral branches from qualifying body nodes. Returns list of new tip ids."""
    if step_idx < LATERAL_BRANCH_MIN_STEP:
        return []

    tips = get_growing_tips(G, root=0)
    if not tips:
        return []

    tips_pos = [node_pos(G, t) for t in tips]
    tip_set = set(tips)

    # Body nodes: non-root, non-tip
    body_nodes = [n for n in G.nodes if n != 0 and n not in tip_set]
    if not body_nodes:
        return []

    new_ids = []
    for body in body_nodes:
        body_pos = node_pos(G, body)

        # Eq. 12: must be at least d from EVERY tip
        dist_to_nearest = min(float(np.linalg.norm(body_pos - tp)) for tp in tips_pos)
        if dist_to_nearest < LATERAL_DISTANCE_FROM_TIP:
            continue

        # Eqs. 10, 11: local N and P at least φ × initial mean
        means = mean_local_NPO(grid, body_pos, DEPLETION_RADIUS)
        N_init = float(NUTRIENT_SPECS["Nitrogen"]["mean"])
        P_init = float(NUTRIENT_SPECS["Phosphorus"]["mean"])
        if (means["Nitrogen"]   < PHI_N * N_init or
            means["Phosphorus"] < PHI_P * P_init):
            continue

        mu_t = growth_rate_from_local(means)
        if mu_t <= 0.0:
            continue

        candidates = top_k_score_voxels(grid, body_pos, DEPLETION_RADIUS, k=None)
        if not candidates:
            continue
        direction = _unit_dir_from_to(body_pos, _voxel_center_um(grid, candidates[0]))
        if direction is None:
            continue

        new_pos = body_pos + mu_t * direction
        idx = grid.world_to_index(*new_pos)
        if idx is None:
            continue
        i, j, k = idx
        if grid.solid_mask[i, j, k] or grid.hypha_mask[i, j, k]:
            continue

        new_id = (max(G.nodes) + 1) if G.number_of_nodes() > 0 else 0
        G.add_node(
            new_id,
            pos=new_pos,
            birth_step=step_idx,
            branch_origin="lateral",
            last_mu=mu_t,
            last_dir=direction,
        )
        G.add_edge(body, new_id)
        grid.hypha_mask[i, j, k] = True
        grid.hypha_owner[i, j, k] = cid
        new_ids.append(new_id)

    return new_ids


# =============================================================================
# PER-TIP DISPATCHER
# =============================================================================

def step_tip(grid: VoxelGrid3D, G: nx.Graph, cid: int, tip_node: int, step_idx: int) -> str:
    """Per-tip dispatch: apical branch (if condition met) else grow.

    Returns one of: "BRANCH", "GROW", "NONE".
    """
    tip_um = node_pos(G, tip_node)
    means = mean_local_NPO(grid, tip_um, DEPLETION_RADIUS)
    mu_t = growth_rate_from_local(means)

    last_mu = G.nodes[tip_node].get("last_mu", None)
    if (step_idx >= APICAL_BRANCH_MIN_STEP
            and last_mu is not None
            and PSI_BRANCH * float(last_mu) >= mu_t):
        new_ids = apical_branch(grid, G, cid, tip_node, step_idx, mu_t)
        if len(new_ids) >= 2:
            return "BRANCH"
        if len(new_ids) == 1:
            return "GROW"  # fallback: only one daughter placed → counts as growth
        # Both daughters failed validation; fall through to ordinary grow attempt.

    new_id = grow_tip(grid, G, cid, tip_node, step_idx, _means=means)
    return "GROW" if new_id is not None else "NONE"


# =============================================================================
# PLOTS
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


def plot_skeletons_3d(graphs: dict, grid: VoxelGrid3D, title: str = "Hypha skeletons"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for cid, G in graphs.items():
        if G.number_of_nodes() == 0:
            continue
        for u, v in G.edges:
            pu = node_pos(G, u); pv = node_pos(G, v)
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], [pu[2], pv[2]],
                    linewidth=1.5, alpha=0.7)
        for n in G.nodes:
            p = node_pos(G, n)
            ax.scatter([p[0]], [p[1]], [p[2]], s=12)
        if 0 in G.nodes:
            p0 = node_pos(G, 0)
            ax.scatter([p0[0]], [p0[1]], [p0[2]], s=80, marker="*", color="k")
            ax.text(p0[0], p0[1], p0[2], f"  cid{cid}", fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("X (µm)"); ax.set_ylabel("Y (µm)"); ax.set_zlabel("Z (µm)")
    ax.set_xlim(grid.min_um[0], grid.max_um[0])
    ax.set_ylim(grid.min_um[1], grid.max_um[1])
    ax.set_zlim(grid.min_um[2], grid.max_um[2])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()


def export_time_series_csv(history: dict):
    n = len(history["total_length_um"])
    rows = []
    for i in range(n):
        row = {
            "step":                 i,
            "time_hr":              float(i * TIME_STEP_HR),
            "total_length_um":      float(history["total_length_um"][i]),
            "total_volume_um3":     float(history["total_volume_um3"][i]),
            "avg_length_um":        float(history["avg_length_um"][i]),
            "avg_volume_um3":       float(history["avg_volume_um3"][i]),
            "avg_tips_per_hypha":   float(history["avg_tips_per_hypha"][i]),
            "n_hyphae":             int(history["n_hyphae"][i]),
            "total_branch_nodes":   int(history["total_branch_nodes"][i]),
            "total_tips":           int(history["total_tips"][i]),
            "citric_total_mmol":    float(history["citric_total_mmol"][i]),
            "citric_mM":            float(history["citric_mM"][i]),
            "liquid_volume_L":      float(history["liquid_volume_L"][i]),
            "apical_branch_events": int(history["apical_branch_events"][i]),
            "lateral_branch_events": int(history["lateral_branch_events"][i]),
        }
        for nutr in NUTRIENT_SPECS.keys():
            row[f"mean_{nutr}_mM"] = float(history["mean_nutrients"][nutr][i])
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["step", "time_hr"]
    write_csv("time_series_all.csv", fieldnames=fieldnames, rows=rows)


def plot_time_series(history: dict):
    """One figure per metric, styled to match the reference plot.

    - Filled-circle data points
    - Smooth same-color fit curve (where applicable)
    - R² shown in the legend
    - Bold title, light gridlines, wider aspect ratio
    """
    export_time_series_csv(history)

    n_steps = len(history["total_length_um"])
    t_hr = np.arange(n_steps) * TIME_STEP_HR

    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    PLOT_COLOR = "#1f77b4"  # matplotlib default blue

    def simple_plot(y, title, ylabel, add_exp_fit=False, clamp_origin=True):
        plt.figure(figsize=(10, 4.5))
        y_arr = np.array(y, dtype=float)

        fit_drawn = False
        if add_exp_fit and n_steps >= 3:
            try:
                popt, _pcov = curve_fit(
                    exponential_func, t_hr, y_arr,
                    p0=[max(abs(y_arr[0]), 1e-6), 0.1, 0.0],
                    maxfev=10000,
                )
                a, b, c = popt
                t_smooth = np.linspace(t_hr[0], t_hr[-1], 200)
                y_smooth = exponential_func(t_smooth, a, b, c)
                y_fit = exponential_func(t_hr, a, b, c)
                ss_res = float(np.sum((y_arr - y_fit) ** 2))
                ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

                plt.plot(t_hr, y_arr, "o", color=PLOT_COLOR, markersize=6)
                plt.plot(
                    t_smooth, y_smooth, "-", color=PLOT_COLOR, linewidth=2,
                    label=f"Fit  (R²={r_squared:.3f})",
                )
                plt.legend(loc="upper left", fontsize=10, framealpha=0.9)
                fit_drawn = True
            except Exception as e:
                print(f"Warning: could not fit exponential to '{title}': {e}")

        if not fit_drawn:
            # Markers + faint connecting line so sparse step counts still read clearly.
            plt.plot(t_hr, y_arr, "o-", color=PLOT_COLOR, markersize=6, linewidth=1)

        plt.title(title, fontweight="bold", fontsize=13)
        plt.xlabel("Time (hrs)")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        if clamp_origin:
            plt.xlim(left=0)
            plt.ylim(bottom=0)
        plt.tight_layout()
        plt.show()

    # Morphology — totals
    simple_plot(history["total_length_um"],   "Total Length",         "Length (µm)",  add_exp_fit=True)
    simple_plot(history["total_volume_um3"],  "Total Hyphal Volume",  "Volume (µm³)", add_exp_fit=True)

    # Morphology — averages per hypha (matches reference y-axis format)
    simple_plot(history["avg_length_um"],      "Average Length Per Hypha",  "Length (µm/no. of hyphae)",  add_exp_fit=True)
    simple_plot(history["avg_volume_um3"],     "Average Volume Per Hypha",  "Volume (µm³/no. of hyphae)", add_exp_fit=True)
    simple_plot(history["avg_tips_per_hypha"], "Average Tips Per Hypha",    "No. of tips/no. of hyphae")

    # Morphology — counts
    simple_plot(history["total_branch_nodes"], "Total Branch Nodes (deg ≥ 3)", "Count")
    simple_plot(history["total_tips"],         "Total Growing Tips",            "Count")

    # Citric acid
    simple_plot(history["citric_mM"],          "Citric Acid Concentration",        "[CA] (mM)")
    simple_plot(history["citric_total_mmol"],  "Cumulative Citric Acid Produced",  "Citric acid (mmol)")

    # Branching events
    simple_plot(history["apical_branch_events"],  "Apical Branching Events",  "Cumulative events")
    simple_plot(history["lateral_branch_events"], "Lateral Branching Events", "Cumulative events")

    # Nutrient time series — auto-scale so small fluctuations near the initial mean stay visible
    for nutr in NUTRIENT_SPECS.keys():
        simple_plot(history["mean_nutrients"][nutr], f"Mean {nutr} in Liquid",
                    "Concentration (mM)", clamp_origin=False)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    _ensure_outdir()

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

    hypha_starts = assign_starting_hyphae(conidia, conidium_radius, seed=RNG_SEED + 2,
                                           origin_um=grid.origin_um)
    graphs, start_dirs = initialize_hypha_graphs(grid, hypha_starts)

    plot_spores_and_initial_dirs(conidia, hypha_starts, conidium_radius)

    # ---- Pre-loop bookkeeping (history, branch counters, snapshot dicts) ----
    history = {
        "total_length_um":      [],
        "total_volume_um3":     [],
        "avg_length_um":        [],
        "avg_volume_um3":       [],
        "avg_tips_per_hypha":   [],
        "n_hyphae":             [],
        "total_branch_nodes":   [],
        "total_tips":           [],
        "citric_total_mmol":    [],
        "citric_mM":            [],
        "liquid_volume_L":      [],
        "mean_nutrients":       {k: [] for k in NUTRIENT_SPECS.keys()},
        "apical_branch_events": [],
        "lateral_branch_events": [],
    }

    branch_event_counters = {"apical": 0, "lateral": 0}

    # Snapshot timesteps for radial / nutrient / citric analyses (20%, 40%, 60%, 80%, 100%)
    snapshot_timesteps = [int(N_STEPS * i / 5) for i in range(1, 6)]
    graph_snapshots = {}      # {timestep: deep_copy_of_graphs}
    nutrient_snapshots = {}   # {timestep: {nutrient_name: {'before': array, 'after': array}}}
    citric_snapshots = {}     # {timestep: {'before': array, 'after': array}}

    # ---- Main loop: branching → consumption → metrics → diffusion ----
    for step in range(N_STEPS):
        # 1. Tip growth + apical branching
        n_grew = 0
        n_apical = 0
        for cid, G in graphs.items():
            if G.number_of_nodes() == 0:
                continue
            tips = get_growing_tips(G, root=0)
            for tip in tips:
                tag = step_tip(grid, G, cid, tip, step)
                if tag == "GROW":
                    n_grew += 1
                elif tag == "BRANCH":
                    n_apical += 1

        # 2. Lateral branching (uses pre-consumption nutrient state)
        n_lateral = 0
        for cid, G in graphs.items():
            if G.number_of_nodes() == 0:
                continue
            n_lateral += len(lateral_branching(grid, G, cid, step))

        branch_event_counters["apical"]  += n_apical
        branch_event_counters["lateral"] += n_lateral

        # 3. Consumption pass: N/P/O + Glucose; CA produced as a by-product
        total_ca_mmol_step, n_ca_events = consume_nutrients(grid, graphs)

        # 4. Record metrics (post-consumption, pre-diffusion — matches Model 3 ordering)
        m = compute_metrics(grid, graphs)
        history["total_length_um"]     .append(m["total_length_um"])
        history["total_volume_um3"]    .append(m["total_volume_um3"])
        history["avg_length_um"]       .append(m["avg_length_um"])
        history["avg_volume_um3"]      .append(m["avg_volume_um3"])
        history["avg_tips_per_hypha"]  .append(m["avg_tips_per_hypha"])
        history["n_hyphae"]            .append(m["n_hyphae"])
        history["total_branch_nodes"]  .append(m["total_branch_nodes"])
        history["total_tips"]          .append(m["total_tips"])
        history["citric_total_mmol"]   .append(m["citric_total_mmol"])
        history["citric_mM"]           .append(m["citric_mM"])
        history["liquid_volume_L"]     .append(m["liquid_volume_L"])
        history["apical_branch_events"] .append(branch_event_counters["apical"])
        history["lateral_branch_events"].append(branch_event_counters["lateral"])

        nut_means = mean_nutrients_over_liquid(grid)
        for nutr in NUTRIENT_SPECS.keys():
            history["mean_nutrients"][nutr].append(nut_means.get(nutr, float("nan")))

        # 5. Diffusion (per-species explicit-Euler Laplacian update)
        diffuse_nutrients(grid)

        print(
            f"step={step}  grew={n_grew}  apical={n_apical}  lateral={n_lateral}  "
            f"ca_events={n_ca_events}  hyphae={m['n_hyphae']}  tips={m['total_tips']}  "
            f"branches={m['total_branch_nodes']}  "
            f"len={m['total_length_um']:.1f}µm  vol={m['total_volume_um3']:.1f}µm³  "
            f"<N>={nut_means['Nitrogen']:.2f} <P>={nut_means['Phosphorus']:.4f} "
            f"<O>={nut_means['Oxygen']:.4g} <G>={nut_means['Glucose']:.2f} mM  "
            f"CA(step)={total_ca_mmol_step:.4g} CA(cum)={m['citric_total_mmol']:.4g} mmol"
        )

    plot_skeletons_3d(graphs, grid, title=f"Hypha skeletons after {N_STEPS} steps")
    plot_time_series(history)
