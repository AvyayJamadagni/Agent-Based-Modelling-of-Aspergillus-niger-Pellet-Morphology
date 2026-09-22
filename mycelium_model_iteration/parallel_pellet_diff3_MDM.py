import os
import re
import csv
import copy
import multiprocessing as mp
from multiprocessing import shared_memory
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — figures are saved to disk, not shown
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# GLOBAL TUNABLES
# =============================================================================

# ---- Output ----
OUTPUT_DIR = "sim_outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SAVE_FIGURES = True   # save figures as PNGs to FIGURES_DIR instead of opening them
FIGURE_DPI = 150

# ---- Grid / domain (µm) ----
EXTENT_UM = 100.0
CONIDIA_SPAWN_BOX_EXTENT_UM = 60.0
N_GRID = 90
RNG_SEED = 13

# Grid visualization (drawing every grid line at N=70 is too heavy)
GRID_PLOT_STRIDE = 5

# ---- Simulation ----
N_STEPS = 20
TIME_STEP_HR = 1.0   # x-axis on time-series plots
N_SHELLS = 70        # concentric shells for hyphal-density / nutrient / citric radial analyses

# ---- Multiprocessing ----
# Number of worker PROCESSES for the parallel consume_nutrients pass.
# Set to 1 to disable the process pool entirely (consume runs serially).
# Nutrient averaging is ALWAYS parallelised across the 4 species using a thread
# pool (no pickling overhead; GIL is released during NumPy random draws).
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# ---- Nutrients (mM) ----
NUTRIENT_SPECS = {
    "Nitrogen":   {"mean": 37.8,   "pm": 1.89},
    "Phosphorus": {"mean": 1.10,   "pm": 0.055},
    "Glucose":    {"mean": 777.10, "pm": 38.855},
    "Oxygen":     {"mean": 0.2109, "pm": 0.010545},
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
MIN_CONIDIA_SEPARATION_UM = 23.5
MAX_PLACEMENT_TRIES = 5000

MISS_HYPHA_BASE_PROB = 0.01
MISS_HYPHA_PER_CONIDIUM = 0.002
MISS_HYPHA_MAX_PROB = 0.10
INITIAL_DIR_CONE_DEG = 90.0   # initial growth direction must lie within this cone of the inward (toward-origin) direction

# ---- Hypha growth ----
MU_MAX = 9.58           # maximum tip growth rate (µm/hr)
DEPLETION_RADIUS = 5.0  # radius (µm) around a tip used for sensing & depletion
HYPHA_RADIUS_UM = 1.4   # physical radius of the hypha cylinder (µm)

# ---- Apical branching ----
PSI_BRANCH = 0.995             # branch when  ψ · μ_{t-1} ≥ μ_t  (drop-in-rate threshold)
MIN_BRANCH_ANGLE_DEG = 50.0   # minimum angle (deg) between the two daughter directions
APICAL_BRANCH_MIN_STEP = 3    # apical branching disabled while step_idx < this

# ---- Lateral branching ----
LATERAL_DISTANCE_FROM_TIP = 4.6   # min distance (µm) from any tip for a lateral branching site
PHI_N = 0.15                      # local N ≥ PHI_N · [N]_init required for a lateral branch
PHI_P = 0.15                      # local P ≥ PHI_P · [P]_init required for a lateral branch
LATERAL_BRANCH_MIN_STEP = 7       # lateral branching disabled while step_idx < this

# ---- Trajectory bias ----
MAX_DEVIATION_DEG = 70.0     # new growth direction can deviate ≤ this from the tip's current direction

# ---- Nutrient averaging (replaces PDE diffusion) ----
# No diffusion coefficients, no PDE, no sub-steps. Each step we redistribute
# every nutrient by re-sampling the same Gaussian field used at initialisation,
# centred on the CURRENT post-consumption liquid mean. See `average_nutrients`.

# ---- Biomass / consumption geometry ----
MYCELIUM_DENSITY_KG_PER_M3 = 150.0   # dry-weight density of mycelium
SUBAPICAL_UPTAKE_FACTOR = 0.1        # subapical (non-tip) uptake = factor × tip MM rate

# ---- Citric acid production ----
EPSILON_N = 0.378                 # local mean N (mM) ≤ this triggers citric acid production
EPSILON_P = 0.0735                # local mean P (mM) ≤ this triggers citric acid production
CITRIC_mmol_per_mol_GLUCOSE = 702.34   # mmol citric acid produced per mol glucose consumed

# ---- Nutrient uptake (Michaelis–Menten kinetics) ----
# Phosphorus
V_MAX_P = 0.08
K_M_P   = 0.0333

# Nitrogen
V_MAX_N = 0.2922
K_M_N   = 0.0735

# Oxygen
V_MAX_O = 12825.0
K_M_O   = 0.0156

# Glucose — dual-pathway transport
V_G1     = 0.00031419
V_G2_MAX = 0.186
K_G2     = 0.26
K_I2     = 933.0


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


def _slug(text: str) -> str:
    """Sanitise a string to a safe filename slug (a-z, 0-9, underscores only)."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text))
    return s.strip("_").lower() or "figure"


def _save_or_show(filename: str):
    """Save the current matplotlib figure to FIGURES_DIR/<filename> (PNG) and close it.

    If SAVE_FIGURES is False, show the figure interactively instead.
    """
    if SAVE_FIGURES:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        if not filename.lower().endswith(".png"):
            filename = filename + ".png"
        full = os.path.join(FIGURES_DIR, filename)
        plt.savefig(full, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        print(f"[FIG] saved: {full}")
    else:
        plt.show()
        plt.close()


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
            "citric_acid_total_mmol": None,
            "citric_acid_spatial_mM": np.zeros((self.n, self.n, self.n), dtype=np.float64),
        }

        # Pre-compute and cache the center meshgrid — fixed for the entire simulation
        # since grid coordinates never change. Avoids rebuilding on every sphere op.
        self._Xc, self._Yc, self._Zc = np.meshgrid(
            self.x_centers, self.y_centers, self.z_centers, indexing="ij"
        )

    def rebuild_center_mesh(self):
        # Returns the cached meshgrid built in __init__.
        return self._Xc, self._Yc, self._Zc

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
    """Apply the tip-elongation rule once. Returns the new node id, or None if no growth."""
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

    chosen_direction = None
    chosen_idx = None
    chosen_pos = None
    for _cand, direction in candidates_dirs:
        new_pos = tip_um + mu * direction
        idx = grid.world_to_index(*new_pos)
        if idx is None:
            continue
        i, j, k = idx
        if grid.solid_mask[i, j, k] or grid.hypha_mask[i, j, k]:
            continue
        chosen_direction = direction
        chosen_idx = (i, j, k)
        chosen_pos = new_pos
        break

    if chosen_direction is None:
        return None

    i, j, k = chosen_idx
    new_id = (max(G.nodes) + 1) if G.number_of_nodes() > 0 else 0
    G.add_node(
        new_id,
        pos=chosen_pos,
        birth_step=step_idx,
        branch_origin="grow",
        last_mu=mu,
        last_dir=chosen_direction,
    )
    G.add_edge(tip_node, new_id)

    grid.hypha_mask[i, j, k] = True
    grid.hypha_owner[i, j, k] = cid
    return new_id


# =============================================================================
# RULE: NUTRIENT CONSUMPTION + CITRIC ACID PRODUCTION
# =============================================================================

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
    """Apply N/P/O/Glucose uptake at a consumption source. Returns mmol citric acid produced."""
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

    for name, (v_max, K_m) in NUTRIENT_MM_PARAMS.items():
        v = michaelis_menten(v_max, K_m, means[name])
        consumed_mmol = v * biomass_g * float(rate_factor)
        if consumed_mmol <= 0.0:
            continue
        delta_mM = (consumed_mmol / n_cells) / voxel_vol_L
        arr = grid.nutrients[name]
        arr[ii, jj, kk] = np.maximum(0.0, arr[ii, jj, kk] - delta_mM)
        grid.nutrients[name] = arr

    G = max(0.0, means["Glucose"])
    ca_active = (means["Nitrogen"] <= EPSILON_N) or (means["Phosphorus"] <= EPSILON_P)

    v_G_passive = V_G1 * G
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

    citric_mmol = 0.0
    if v_G_active > 0.0:
        active_glucose_mmol = v_G_active * biomass_g * float(rate_factor)
        citric_mmol = active_glucose_mmol * CITRIC_mmol_per_mol_GLUCOSE / 1000.0

        if cid is not None:
            grid.metabolites["citric_acid_total_mmol"][cid] += citric_mmol

        delta_C_mM = (citric_mmol / n_cells) / voxel_vol_L
        grid.metabolites["citric_acid_spatial_mM"][ii, jj, kk] += delta_C_mM

    return citric_mmol


def consume_nutrients(grid: VoxelGrid3D, graphs: dict):
    """Serial all-nutrient consumption pass. Returns (total_ca_mmol, n_ca_events)."""
    tip_biomass = cylinder_biomass_g(DEPLETION_RADIUS, HYPHA_RADIUS_UM)

    total_ca_mmol = 0.0
    n_ca_events = 0

    for cid, G in graphs.items():
        if G.number_of_nodes() == 0:
            continue
        for tip in get_growing_tips(G, root=0):
            ca = consume_at_point(grid, node_pos(G, tip), tip_biomass,
                                  rate_factor=1.0, cid=cid)
            total_ca_mmol += ca
            if ca > 0.0:
                n_ca_events += 1

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
# PARALLEL CONSUME PASS (multiprocessing + shared memory)
# =============================================================================
#
# Strategy
# --------
# `consume_nutrients` is the dominant per-step cost. We parallelise it by:
#   1. Building a flat list of "consumption tasks" (one per tip + one per edge).
#   2. Sharing the current grid arrays via multiprocessing.shared_memory so
#      worker processes can read them without pickling large arrays per task.
#   3. Each worker computes the SPARSE delta its task applies (just the voxel
#      indices and per-voxel Δ values), and returns it.
#   4. The main process merges all sparse deltas into the grid serially.
#
# Every consumer sees the START-of-pass state (the snapshot pushed into shared
# memory). Their deltas are merged with a clamp-at-0 guard. For our scale this
# is a fine approximation and the only way to expose meaningful task-level
# parallelism in Python without per-voxel locks.

# ---- Worker process globals (populated once per worker by _consume_worker_init) ----
_W_NUTRIENTS = None
_W_CITRIC = None
_W_HYPHA_MASK = None
_W_SOLID_MASK = None
_W_XC = None
_W_YC = None
_W_ZC = None
_W_VOXEL_VOL_L = None
_W_DEPLETION_RADIUS = None
_W_HYPHA_RADIUS_UM = None
_W_NUT_MM_PARAMS = None
_W_EPSILON_N = None
_W_EPSILON_P = None
_W_V_G1 = None
_W_V_G2_MAX = None
_W_K_G2 = None
_W_K_I2 = None
_W_LAMBDA_CA = None
_W_SUBAPICAL_FACTOR = None
_W_SHM_REFS = None  # keep SharedMemory objects alive in this worker


def _consume_worker_init(metadata):
    """Run once per worker process. Attach shared memory and cache static state."""
    global _W_NUTRIENTS, _W_CITRIC, _W_HYPHA_MASK, _W_SOLID_MASK
    global _W_XC, _W_YC, _W_ZC
    global _W_VOXEL_VOL_L, _W_DEPLETION_RADIUS, _W_HYPHA_RADIUS_UM
    global _W_NUT_MM_PARAMS, _W_EPSILON_N, _W_EPSILON_P
    global _W_V_G1, _W_V_G2_MAX, _W_K_G2, _W_K_I2
    global _W_LAMBDA_CA, _W_SUBAPICAL_FACTOR
    global _W_SHM_REFS

    _W_SHM_REFS = []

    _W_NUTRIENTS = {}
    for name, (shm_name, shape, dtype_str) in metadata["nutrients"].items():
        shm = shared_memory.SharedMemory(name=shm_name)
        _W_SHM_REFS.append(shm)
        _W_NUTRIENTS[name] = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)

    citric_name, citric_shape, citric_dtype = metadata["citric"]
    shm = shared_memory.SharedMemory(name=citric_name)
    _W_SHM_REFS.append(shm)
    _W_CITRIC = np.ndarray(citric_shape, dtype=np.dtype(citric_dtype), buffer=shm.buf)

    h_name, h_shape, h_dtype = metadata["hypha_mask"]
    shm = shared_memory.SharedMemory(name=h_name)
    _W_SHM_REFS.append(shm)
    _W_HYPHA_MASK = np.ndarray(h_shape, dtype=np.dtype(h_dtype), buffer=shm.buf)

    s_name, s_shape, s_dtype = metadata["solid_mask"]
    shm = shared_memory.SharedMemory(name=s_name)
    _W_SHM_REFS.append(shm)
    _W_SOLID_MASK = np.ndarray(s_shape, dtype=np.dtype(s_dtype), buffer=shm.buf)

    x_centers = metadata["x_centers"]
    y_centers = metadata["y_centers"]
    z_centers = metadata["z_centers"]
    _W_XC, _W_YC, _W_ZC = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")

    _W_VOXEL_VOL_L      = float(metadata["voxel_vol_L"])
    _W_DEPLETION_RADIUS = float(metadata["depletion_radius"])
    _W_HYPHA_RADIUS_UM  = float(metadata["hypha_radius_um"])
    _W_NUT_MM_PARAMS    = metadata["nut_mm_params"]
    _W_EPSILON_N        = float(metadata["epsilon_n"])
    _W_EPSILON_P        = float(metadata["epsilon_p"])
    _W_V_G1             = float(metadata["v_g1"])
    _W_V_G2_MAX         = float(metadata["v_g2_max"])
    _W_K_G2             = float(metadata["k_g2"])
    _W_K_I2             = float(metadata["k_i2"])
    _W_LAMBDA_CA        = float(metadata["lambda_ca"])
    _W_SUBAPICAL_FACTOR = float(metadata["subapical_factor"])


def _consume_worker_compute(task):
    """Compute one consumption task. Returns sparse delta or None.

    Task tuple: (cid, point_um (3,), biomass_g, rate_factor)
    Return: (cid, voxel_idx (k,3), nut_deltas_mM dict, citric_mmol, citric_delta_mM)
    """
    cid, point_um, biomass_g, rate_factor = task

    dx = _W_XC - point_um[0]
    dy = _W_YC - point_um[1]
    dz = _W_ZC - point_um[2]
    sphere = (dx * dx + dy * dy + dz * dz) <= (_W_DEPLETION_RADIUS * _W_DEPLETION_RADIUS)
    liquid = sphere & (~_W_SOLID_MASK) & (~_W_HYPHA_MASK)
    cand_idx = np.argwhere(liquid)
    if cand_idx.size == 0:
        return None

    n_cells = int(cand_idx.shape[0])

    means = {}
    for name in ("Nitrogen", "Phosphorus", "Oxygen", "Glucose"):
        arr = _W_NUTRIENTS[name]
        valid = liquid & np.isfinite(arr)
        means[name] = float(np.mean(arr[valid])) if np.any(valid) else 0.0

    nut_deltas = {}

    for name in ("Nitrogen", "Phosphorus", "Oxygen"):
        v_max, K_m = _W_NUT_MM_PARAMS[name]
        S = max(0.0, means[name])
        denom = K_m + S
        v = (v_max * S / denom) if denom > 0.0 else 0.0
        consumed_mmol = v * biomass_g * float(rate_factor)
        if consumed_mmol > 0.0:
            nut_deltas[name] = (consumed_mmol / n_cells) / _W_VOXEL_VOL_L

    G = max(0.0, means["Glucose"])
    ca_active = (means["Nitrogen"] <= _W_EPSILON_N) or (means["Phosphorus"] <= _W_EPSILON_P)

    v_G_passive = _W_V_G1 * G
    v_G_active = 0.0
    if ca_active and G > 0.0:
        citric_local = float(np.mean(_W_CITRIC[liquid])) if np.any(liquid) else 0.0
        C = max(0.0, citric_local)
        inhibition = 1.0 + (C / _W_K_I2)
        denom = (_W_K_G2 + G) * inhibition
        if denom > 0.0:
            v_G_active = _W_V_G2_MAX * G / denom

    glucose_consumed_mmol = (v_G_passive + v_G_active) * biomass_g * float(rate_factor)
    if glucose_consumed_mmol > 0.0:
        nut_deltas["Glucose"] = (glucose_consumed_mmol / n_cells) / _W_VOXEL_VOL_L

    citric_mmol = 0.0
    citric_delta_mM = 0.0
    if v_G_active > 0.0:
        active_glucose_mmol = v_G_active * biomass_g * float(rate_factor)
        citric_mmol = active_glucose_mmol * _W_LAMBDA_CA / 1000.0
        citric_delta_mM = (citric_mmol / n_cells) / _W_VOXEL_VOL_L

    return (cid, cand_idx, nut_deltas, citric_mmol, citric_delta_mM)


class _SharedGridState:
    """Manages shared-memory mirrors of the dynamic grid arrays workers need.

    Lifetime:
      - Created once before the simulation loop.
      - `sync_from(grid)` called at the start of each parallel consume pass.
      - `cleanup()` must be called before exit (close + unlink).
    """

    def __init__(self, grid: "VoxelGrid3D"):
        self._shms = {}
        self._views = {}

        def _make(name, src):
            shm = shared_memory.SharedMemory(create=True, size=src.nbytes)
            view = np.ndarray(src.shape, dtype=src.dtype, buffer=shm.buf)
            view[:] = src
            self._shms[name] = shm
            self._views[name] = view
            return (shm.name, src.shape, str(src.dtype))

        nut_meta = {}
        for name, arr in grid.nutrients.items():
            nut_meta[name] = _make(f"nut_{name}", arr)

        citric_meta = _make("citric", grid.metabolites["citric_acid_spatial_mM"])
        hypha_meta  = _make("hypha_mask", grid.hypha_mask)
        solid_meta  = _make("solid_mask", grid.solid_mask)

        self.metadata = {
            "nutrients":        nut_meta,
            "citric":           citric_meta,
            "hypha_mask":       hypha_meta,
            "solid_mask":       solid_meta,
            "x_centers":        np.asarray(grid.x_centers, dtype=float),
            "y_centers":        np.asarray(grid.y_centers, dtype=float),
            "z_centers":        np.asarray(grid.z_centers, dtype=float),
            "voxel_vol_L":      grid.voxel_volume_L(),
            "depletion_radius": DEPLETION_RADIUS,
            "hypha_radius_um":  HYPHA_RADIUS_UM,
            "nut_mm_params":    dict(NUTRIENT_MM_PARAMS),
            "epsilon_n":        EPSILON_N,
            "epsilon_p":        EPSILON_P,
            "v_g1":             V_G1,
            "v_g2_max":         V_G2_MAX,
            "k_g2":             K_G2,
            "k_i2":             K_I2,
            "lambda_ca":        CITRIC_mmol_per_mol_GLUCOSE,
            "subapical_factor": SUBAPICAL_UPTAKE_FACTOR,
        }

    def sync_from(self, grid: "VoxelGrid3D"):
        """Copy current grid state into the shared-memory views."""
        for name, arr in grid.nutrients.items():
            self._views[f"nut_{name}"][:] = arr
        self._views["citric"][:]     = grid.metabolites["citric_acid_spatial_mM"]
        self._views["hypha_mask"][:] = grid.hypha_mask
        self._views["solid_mask"][:] = grid.solid_mask

    def cleanup(self):
        """Close and unlink all shared-memory blocks."""
        for shm in self._shms.values():
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        self._shms.clear()
        self._views.clear()


def _build_consumption_tasks(graphs):
    """Flatten tips and edges into a list of (cid, point_um, biomass_g, rate_factor) tasks."""
    tip_biomass = cylinder_biomass_g(DEPLETION_RADIUS, HYPHA_RADIUS_UM)
    tasks = []

    for cid, G in graphs.items():
        if G.number_of_nodes() == 0:
            continue
        for tip in get_growing_tips(G, root=0):
            tasks.append((cid, np.asarray(node_pos(G, tip), dtype=float),
                          tip_biomass, 1.0))

    for cid, G in graphs.items():
        for u, v in G.edges:
            pu = node_pos(G, u)
            pv = node_pos(G, v)
            edge_length = float(np.linalg.norm(pv - pu))
            if edge_length < 1e-12:
                continue
            edge_midpoint = 0.5 * (pu + pv)
            edge_biomass = cylinder_biomass_g(edge_length, HYPHA_RADIUS_UM)
            tasks.append((cid, np.asarray(edge_midpoint, dtype=float),
                          edge_biomass, SUBAPICAL_UPTAKE_FACTOR))

    return tasks


def _apply_consume_results(grid: VoxelGrid3D, results):
    """Merge sparse worker deltas back into the grid. Returns (total_ca_mmol, n_ca_events)."""
    total_ca_mmol = 0.0
    n_ca_events = 0

    for result in results:
        if result is None:
            continue
        cid, cand_idx, nut_deltas, citric_mmol, citric_delta_mM = result
        ii = cand_idx[:, 0]
        jj = cand_idx[:, 1]
        kk = cand_idx[:, 2]

        for name, delta_mM in nut_deltas.items():
            arr = grid.nutrients[name]
            arr[ii, jj, kk] = np.maximum(0.0, arr[ii, jj, kk] - delta_mM)
            grid.nutrients[name] = arr

        if citric_mmol > 0.0:
            grid.metabolites["citric_acid_total_mmol"][cid] += citric_mmol
            grid.metabolites["citric_acid_spatial_mM"][ii, jj, kk] += citric_delta_mM
            total_ca_mmol += citric_mmol
            n_ca_events += 1

    return total_ca_mmol, n_ca_events


def consume_nutrients_parallel(grid: VoxelGrid3D, graphs: dict, pool, shared_state: _SharedGridState):
    """Parallel version of `consume_nutrients`. Falls back to serial if pool is None."""
    tasks = _build_consumption_tasks(graphs)
    if not tasks:
        return 0.0, 0

    if pool is None:
        return consume_nutrients(grid, graphs)

    # Push current grid state into shared memory so workers see fresh data.
    shared_state.sync_from(grid)
    results = pool.map(_consume_worker_compute, tasks)
    return _apply_consume_results(grid, results)


# =============================================================================
# NUTRIENT AVERAGING (replacement for PDE diffusion)
# =============================================================================
#
# Instead of solving ∂C/∂t = D∇²C numerically, this models the liquid medium as
# a perfectly-mixed bath that re-equilibrates between simulation steps. After
# consumption has lowered the total mass of each nutrient, we:
#
#   1. Compute the mean concentration of each species over the CURRENT liquid
#      voxels (post-consumption, so slightly less than the previous step).
#   2. Re-sample a fresh Gaussian field at that new mean with the SAME ±pm
#      spread used at initialisation.
#   3. Write the new field into liquid voxels only.
#
# Serial path: `average_nutrients` — uses a module-level RNG for reproducibility.
# Parallel path: `average_nutrients_parallel` — resamples all 4 species
#   SIMULTANEOUSLY using a ThreadPoolExecutor. Each species gets a deterministic
#   seed derived from (step_idx, species_index) so results are fully reproducible.
#   NumPy releases the GIL during random-field generation, so threads genuinely
#   overlap in time with no contention. No pickling overhead because threads
#   share the same process memory.

_NUTRIENT_AVG_RNG = np.random.default_rng(RNG_SEED + 1)
_NUTRIENT_AVG_BASE_SEED = RNG_SEED + 1   # base for parallel path seeds


def average_nutrients(grid: VoxelGrid3D):
    """Serial: redistribute each nutrient as a fresh Gaussian field at the post-consumption mean."""
    liquid = (~grid.solid_mask) & (~grid.hypha_mask)
    if not np.any(liquid):
        return

    for name, spec in NUTRIENT_SPECS.items():
        arr = grid.nutrients.get(name)
        if arr is None:
            continue

        liquid_vals = arr[liquid]
        finite = np.isfinite(liquid_vals)
        if not np.any(finite):
            continue
        new_mean = float(np.mean(liquid_vals[finite]))
        if new_mean < 0.0:
            new_mean = 0.0

        pm = float(spec["pm"])
        sigma = pm * float(SIGMA_SCALE[name])
        clip = pm if CLAMP_TO_PM_RANGE else None
        new_field = init_gaussian_field(grid.n, new_mean, sigma, clip,
                                        rng=_NUTRIENT_AVG_RNG)

        out = arr.copy()
        out[liquid] = new_field[liquid]
        out[grid.hypha_mask] = 0.0
        out[grid.solid_mask] = np.nan
        grid.nutrients[name] = out


def _avg_resample_worker(n, new_mean, sigma, clip_half_range, seed):
    """Pure function: generate a Gaussian nutrient field at new_mean.

    Called from a thread pool — creates its own local RNG so threads have
    no shared state. NumPy releases the GIL during rng.normal(), enabling
    genuine parallel execution across the 4 nutrient species.
    """
    rng = np.random.default_rng(seed)
    field = new_mean + rng.normal(0.0, sigma, size=(n, n, n))
    if clip_half_range is not None:
        lo, hi = new_mean - clip_half_range, new_mean + clip_half_range
        field = np.clip(field, lo, hi)
    field = field - field.mean() + new_mean
    if clip_half_range is not None:
        field = np.clip(field, lo, hi)
        field = field - field.mean() + new_mean
    return field.astype(np.float32)


def average_nutrients_parallel(grid: VoxelGrid3D, step_idx: int):
    """Parallel: resample all nutrient species simultaneously with a thread pool.

    Seeds are deterministic:
        seed = _NUTRIENT_AVG_BASE_SEED + step_idx * N_NUTRIENTS + nutrient_index
    This guarantees full reproducibility while keeping species independent.
    Thread count is capped at min(N_WORKERS, number_of_nutrients) — using more
    threads than nutrients provides no benefit.
    """
    liquid = (~grid.solid_mask) & (~grid.hypha_mask)
    if not np.any(liquid):
        return

    # Pre-compute new means serially (fast numpy reductions).
    n_nutrients = len(NUTRIENT_SPECS)
    tasks = []  # (name, new_mean, sigma, clip, seed)
    for nut_idx, (name, spec) in enumerate(NUTRIENT_SPECS.items()):
        arr = grid.nutrients.get(name)
        if arr is None:
            continue
        liquid_vals = arr[liquid]
        finite = np.isfinite(liquid_vals)
        if not np.any(finite):
            continue
        new_mean = max(0.0, float(np.mean(liquid_vals[finite])))
        pm = float(spec["pm"])
        sigma = pm * float(SIGMA_SCALE[name])
        clip = pm if CLAMP_TO_PM_RANGE else None
        seed = _NUTRIENT_AVG_BASE_SEED + step_idx * n_nutrients + nut_idx
        tasks.append((name, new_mean, sigma, clip, seed))

    if not tasks:
        return

    # Resample fields in parallel — threads share memory, zero pickling cost.
    n_threads = min(N_WORKERS, len(tasks))
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            (name, executor.submit(_avg_resample_worker, grid.n, new_mean, sigma, clip, seed))
            for name, new_mean, sigma, clip, seed in tasks
        ]
        new_fields = {name: fut.result() for name, fut in futures}

    # Apply results to the grid serially (no data races on grid.nutrients).
    for name, new_field in new_fields.items():
        out = grid.nutrients[name].copy()
        out[liquid] = new_field[liquid]
        out[grid.hypha_mask] = 0.0
        out[grid.solid_mask] = np.nan
        grid.nutrients[name] = out


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

def apical_branch(grid: VoxelGrid3D, G: nx.Graph, cid: int, tip_node: int,
                  step_idx: int, mu_t: float) -> list:
    """Create up to 2 apical daughter tips. Returns list of newly added node ids."""
    tip_um = node_pos(G, tip_node)
    candidates = top_k_score_voxels(grid, tip_um, DEPLETION_RADIUS, k=None)
    if not candidates:
        return []

    prev_dir = G.nodes[tip_node].get("last_dir", None)
    candidates_dirs = _candidates_in_trajectory_cone(grid, tip_um, prev_dir, candidates)
    if not candidates_dirs:
        return []

    first_dir = candidates_dirs[0][1]

    cos_branch = float(np.cos(np.deg2rad(MIN_BRANCH_ANGLE_DEG)))
    second_dir = None
    for _cand, d in candidates_dirs[1:]:
        if float(np.dot(first_dir, d)) <= cos_branch:
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

def lateral_branching(grid: VoxelGrid3D, G: nx.Graph, cid: int, step_idx: int) -> list:
    """Spawn lateral branches from qualifying body nodes. Returns list of new tip ids."""
    if step_idx < LATERAL_BRANCH_MIN_STEP:
        return []

    tips = get_growing_tips(G, root=0)
    if not tips:
        return []

    tips_pos = [node_pos(G, t) for t in tips]
    tip_set = set(tips)

    body_nodes = [n for n in G.nodes if n != 0 and n not in tip_set]
    if not body_nodes:
        return []

    new_ids = []
    for body in body_nodes:
        body_pos = node_pos(G, body)

        dist_to_nearest = min(float(np.linalg.norm(body_pos - tp)) for tp in tips_pos)
        if dist_to_nearest < LATERAL_DISTANCE_FROM_TIP:
            continue

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
            return "GROW"

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

    ax.set_title("Cartesian axes + 3D voxel grid (subsampled)", fontsize=20, fontweight="bold")
    ax.set_xlabel("X (µm)", fontsize=18)
    ax.set_ylabel("Y (µm)", fontsize=18)
    ax.set_zlabel("Z (µm)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(grid.min_um[0], grid.max_um[0])
    ax.set_ylim(grid.min_um[1], grid.max_um[1])
    ax.set_zlim(grid.min_um[2], grid.max_um[2])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    _save_or_show("axes_grid.png")

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

    ax.set_title("Spore locations + initial hypha direction(s)", fontsize=20, fontweight="bold")
    ax.set_xlabel("X (µm)", fontsize=18)
    ax.set_ylabel("Y (µm)", fontsize=18)
    ax.set_zlabel("Z (µm)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    half = EXTENT_UM / 2.0
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    _save_or_show("spores_initial_dirs.png")

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

    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.set_xlabel("X (µm)", fontsize=18)
    ax.set_ylabel("Y (µm)", fontsize=18)
    ax.set_zlabel("Z (µm)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(grid.min_um[0], grid.max_um[0])
    ax.set_ylim(grid.min_um[1], grid.max_um[1])
    ax.set_zlim(grid.min_um[2], grid.max_um[2])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    _save_or_show(f"skeletons_3d_{_slug(title)}.png")


def export_time_series_csv(history: dict):
    n = len(history["total_length_um"])
    rows = []
    for i in range(n):
        row = {
            "step":                  i,
            "time_hr":               float(i * TIME_STEP_HR),
            "total_length_um":       float(history["total_length_um"][i]),
            "total_volume_um3":      float(history["total_volume_um3"][i]),
            "avg_length_um":         float(history["avg_length_um"][i]),
            "avg_volume_um3":        float(history["avg_volume_um3"][i]),
            "avg_tips_per_hypha":    float(history["avg_tips_per_hypha"][i]),
            "n_hyphae":              int(history["n_hyphae"][i]),
            "total_branch_nodes":    int(history["total_branch_nodes"][i]),
            "total_tips":            int(history["total_tips"][i]),
            "citric_total_mmol":     float(history["citric_total_mmol"][i]),
            "citric_mM":             float(history["citric_mM"][i]),
            "liquid_volume_L":       float(history["liquid_volume_L"][i]),
            "apical_branch_events":  int(history["apical_branch_events"][i]),
            "lateral_branch_events": int(history["lateral_branch_events"][i]),
        }
        for nutr in NUTRIENT_SPECS.keys():
            row[f"mean_{nutr}_mM"] = float(history["mean_nutrients"][nutr][i])
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["step", "time_hr"]
    write_csv("time_series_all.csv", fieldnames=fieldnames, rows=rows)


def plot_time_series(history: dict):
    """One figure per metric, styled with bold titles and larger fonts."""
    export_time_series_csv(history)

    n_steps = len(history["total_length_um"])
    t_hr = np.arange(n_steps) * TIME_STEP_HR

    def exponential_func(x, a, b, c):
        return a * np.exp(b * x) + c

    PLOT_COLOR = "#1f77b4"

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
                plt.legend(loc="upper left", fontsize=16, framealpha=0.9)
                fit_drawn = True
            except Exception as e:
                print(f"Warning: could not fit exponential to '{title}': {e}")

        if not fit_drawn:
            plt.plot(t_hr, y_arr, "o-", color=PLOT_COLOR, markersize=6, linewidth=1)

        plt.title(title, fontweight="bold", fontsize=20)
        plt.xlabel("Time (hrs)", fontsize=18)
        plt.ylabel(ylabel, fontsize=18)
        plt.tick_params(axis="both", labelsize=16)
        plt.grid(True, alpha=0.3)
        if clamp_origin:
            plt.xlim(left=0)
            plt.ylim(bottom=0)
        plt.tight_layout()
        _save_or_show(f"timeseries_{_slug(title)}.png")

    simple_plot(history["total_length_um"],   "Total Length",         "Length (µm)",  add_exp_fit=True)
    simple_plot(history["total_volume_um3"],  "Total Hyphal Volume",  "Volume (µm³)", add_exp_fit=True)

    simple_plot(history["avg_length_um"],      "Average Length Per Hypha",  "Length (µm/no. of hyphae)",  add_exp_fit=True)
    simple_plot(history["avg_volume_um3"],     "Average Volume Per Hypha",  "Volume (µm³/no. of hyphae)", add_exp_fit=True)
    simple_plot(history["avg_tips_per_hypha"], "Average Tips Per Hypha",    "No. of tips/no. of hyphae")

    simple_plot(history["total_branch_nodes"], "Total Branch Nodes (deg ≥ 3)", "Count")
    simple_plot(history["total_tips"],         "Total Growing Tips",            "Count")

    simple_plot(history["citric_mM"],          "Citric Acid Concentration",        "[CA] (mM)")
    simple_plot(history["citric_total_mmol"],  "Cumulative Citric Acid Produced",  "Citric acid (mmol)")

    simple_plot(history["apical_branch_events"],  "Apical Branching Events",  "Cumulative events")
    simple_plot(history["lateral_branch_events"], "Lateral Branching Events", "Cumulative events")

    for nutr in NUTRIENT_SPECS.keys():
        simple_plot(history["mean_nutrients"][nutr], f"Mean {nutr} in Liquid",
                    "Concentration (mM)", clamp_origin=False)


# =============================================================================
# SHELL-BASED RADIAL ANALYSIS
# =============================================================================
#
# Pellet center is the centroid of the conidia. Space is divided into N_SHELLS
# concentric spherical shells out to the furthest hypha node from the FINAL
# graph state — every snapshot reuses the same bins for fair comparison.
#
# Three per-snapshot analyses:
#   1. Hyphal density   — fraction of each shell's volume occupied by hyphae.
#   2. Nutrient delta   — voxel-mean (before − after) per shell, per species.
#   3. Citric production — total mmol produced per shell during that step.

def compute_shell_boundaries(center_um, graphs, n_shells=N_SHELLS):
    """Find max distance from center across all nodes; return (max_radius, shell_radii)."""
    max_dist = 0.0
    for G in graphs.values():
        for n in G.nodes:
            pos = node_pos(G, n)
            dist = float(np.linalg.norm(pos - center_um))
            max_dist = max(max_dist, dist)
    if max_dist == 0.0:
        max_dist = 1.0
    shell_radii = np.linspace(0, max_dist, n_shells + 1)
    return max_dist, shell_radii


def edge_length_in_shell(p1, p2, center, r_inner, r_outer):
    """Length of edge segment (p1, p2) inside the spherical shell [r_inner, r_outer]."""
    p1, p2, center = np.array(p1), np.array(p2), np.array(center)
    edge_vec = p2 - p1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-12:
        return 0.0
    n_samples = 100
    t_vals = np.linspace(0, 1, n_samples)
    sample_points = p1[np.newaxis, :] + t_vals[:, np.newaxis] * edge_vec[np.newaxis, :]
    dists = np.linalg.norm(sample_points - center[np.newaxis, :], axis=1)
    in_shell = (dists >= r_inner) & (dists <= r_outer)
    fraction_in_shell = np.sum(in_shell) / n_samples
    return float(edge_len * fraction_in_shell)


def analyze_hyphal_density_by_shell(center_um, graphs, n_shells=N_SHELLS):
    """Compute hyphal length / volume / fraction per concentric shell."""
    max_radius, shell_radii = compute_shell_boundaries(center_um, graphs, n_shells)

    results = {
        'shell_radii':       shell_radii,
        'shell_inner':       shell_radii[:-1],
        'shell_outer':       shell_radii[1:],
        'shell_mid':         (shell_radii[:-1] + shell_radii[1:]) / 2,
        'shell_volumes':     [],
        'hyphal_lengths':    [],
        'hyphal_volumes':    [],
        'hyphal_fractions':  [],
        'edges_per_shell':   [],
    }

    for i in range(n_shells):
        r_inner = float(shell_radii[i])
        r_outer = float(shell_radii[i + 1])

        shell_vol = (4.0 / 3.0) * np.pi * (r_outer ** 3 - r_inner ** 3)
        results['shell_volumes'].append(shell_vol)

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

        hyphal_vol = total_length_in_shell * np.pi * (HYPHA_RADIUS_UM ** 2)
        results['hyphal_volumes'].append(hyphal_vol)

        hyphal_frac = (hyphal_vol / shell_vol) if shell_vol > 0 else 0.0
        results['hyphal_fractions'].append(hyphal_frac)

    return results


def analyze_nutrient_by_shell(center_um, grid, nutrient_before, nutrient_after,
                              nutrient_name, shell_radii):
    """Voxel-mean of (before, after, delta) per shell over LIQUID voxels (mM)."""
    n_shells = len(shell_radii) - 1

    results = {
        'shell_inner':         shell_radii[:-1],
        'shell_outer':         shell_radii[1:],
        'shell_mid':           (shell_radii[:-1] + shell_radii[1:]) / 2,
        'nutrient_before_avg': [],
        'nutrient_after_avg':  [],
        'nutrient_delta_avg':  [],
    }

    Xc, Yc, Zc = grid.rebuild_center_mesh()
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    distances = np.sqrt(dx * dx + dy * dy + dz * dz)

    for i in range(n_shells):
        r_inner = float(shell_radii[i])
        r_outer = float(shell_radii[i + 1])

        in_shell = (distances >= r_inner) & (distances < r_outer)
        liquid_in_shell = in_shell & (~grid.solid_mask) & (~grid.hypha_mask)

        if np.any(liquid_in_shell):
            before_vals = nutrient_before[liquid_in_shell]
            after_vals  = nutrient_after[liquid_in_shell]
            valid = np.isfinite(before_vals) & np.isfinite(after_vals)
            if np.any(valid):
                avg_before = float(np.mean(before_vals[valid]))
                avg_after  = float(np.mean(after_vals[valid]))
                avg_delta  = avg_before - avg_after
            else:
                avg_before = avg_after = avg_delta = 0.0
        else:
            avg_before = avg_after = avg_delta = 0.0

        results['nutrient_before_avg'].append(avg_before)
        results['nutrient_after_avg'].append(avg_after)
        results['nutrient_delta_avg'].append(avg_delta)

    return results


def analyze_citric_acid_by_shell(center_um, grid, citric_spatial_before,
                                 citric_spatial_after, shell_radii):
    """Total mmol citric acid produced per shell during the captured step."""
    n_shells = len(shell_radii) - 1

    results = {
        'shell_inner':            shell_radii[:-1],
        'shell_outer':            shell_radii[1:],
        'shell_mid':              (shell_radii[:-1] + shell_radii[1:]) / 2,
        'citric_production_mmol': [],
    }

    Xc, Yc, Zc = grid.rebuild_center_mesh()
    dx = Xc - center_um[0]
    dy = Yc - center_um[1]
    dz = Zc - center_um[2]
    distances = np.sqrt(dx * dx + dy * dy + dz * dz)

    citric_delta_mM = citric_spatial_after - citric_spatial_before
    voxel_vol_L = grid.voxel_volume_L()

    for i in range(n_shells):
        r_inner = float(shell_radii[i])
        r_outer = float(shell_radii[i + 1])
        in_shell = (distances >= r_inner) & (distances < r_outer)
        total_mmol = float(np.sum(citric_delta_mM[in_shell]) * voxel_vol_L) if np.any(in_shell) else 0.0
        results['citric_production_mmol'].append(total_mmol)

    return results


def plot_shell_cross_section(center_um, graphs, shell_data, highlight_shell_idx=None):
    """XY-plane cross-section with concentric shell circles."""
    fig, ax = plt.subplots(figsize=(10, 10))

    shell_radii = shell_data['shell_radii']
    for r in shell_radii:
        circle = plt.Circle((center_um[0], center_um[1]), r, fill=False,
                            edgecolor='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.add_patch(circle)

    if highlight_shell_idx is not None:
        r_inner = shell_data['shell_inner'][highlight_shell_idx]
        r_outer = shell_data['shell_outer'][highlight_shell_idx]
        for r in [r_inner, r_outer]:
            circle = plt.Circle((center_um[0], center_um[1]), r, fill=False,
                                edgecolor='blue', linewidth=3, alpha=0.8)
            ax.add_patch(circle)

    for _cid, G in graphs.items():
        for u, v in G.edges:
            pu = node_pos(G, u)
            pv = node_pos(G, v)
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], 'k-', linewidth=0.8, alpha=0.3)

    if highlight_shell_idx is not None:
        edges_in_shell = shell_data['edges_per_shell'][highlight_shell_idx]
        for _cid, _u, _v, pu, pv, _length in edges_in_shell:
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], 'r-', linewidth=2.5, alpha=0.8)

    ax.plot(center_um[0], center_um[1], 'k*', markersize=15, label='Pellet center')
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)', fontsize=18)
    ax.set_ylabel('Y (µm)', fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    if highlight_shell_idx is not None:
        ax.set_title(f'Cross-section with shell {highlight_shell_idx} highlighted',
                     fontsize=20, fontweight="bold")
        fig_name = f"shell_cross_section_shell{highlight_shell_idx}.png"
    else:
        ax.set_title('Cross-section showing all shells', fontsize=20, fontweight="bold")
        fig_name = "shell_cross_section_all.png"
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig_name)


def _radial_curve_with_inner_avg(ax, distances, values, color, label, inner_cutoff_um):
    """Plot a radial curve, collapsing inner shells to a single mean."""
    distances = np.array(distances)
    values = np.array(values)

    mask_inner = distances <= inner_cutoff_um
    avg_inner = float(np.mean(values[mask_inner])) if np.any(mask_inner) else None
    if avg_inner is not None:
        ax.plot([0, inner_cutoff_um], [avg_inner, avg_inner], '-', linewidth=3, color=color)

    mask_outer = distances > inner_cutoff_um
    if np.any(mask_outer):
        d_plot = distances[mask_outer]
        v_plot = values[mask_outer]
        if avg_inner is not None:
            d_plot = np.concatenate([[inner_cutoff_um], d_plot])
            v_plot = np.concatenate([[avg_inner], v_plot])
        ax.plot(d_plot, v_plot, 'o-', linewidth=3, markersize=7, color=color, label=label)
    elif avg_inner is not None:
        ax.plot([], [], 'o-', linewidth=3, markersize=7, color=color, label=label)


def plot_hyphal_density_analysis(shell_data_list, timesteps_list):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(shell_data_list)))
    inner_cutoff = CONIDIA_SPAWN_BOX_EXTENT_UM / 2 * np.sqrt(3)

    for i, (shell_data, step) in enumerate(zip(shell_data_list, timesteps_list)):
        time_hr = step * TIME_STEP_HR
        _radial_curve_with_inner_avg(
            ax, shell_data['shell_mid'], shell_data['hyphal_fractions'],
            colors[i], f'Step {step} ({time_hr:.1f} hr)', inner_cutoff,
        )

    ax.set_xlabel('Distance from center (µm)', fontsize=18)
    ax.set_ylabel('Hyphal fraction (dimensionless)', fontsize=18)
    ax.set_title('Hyphal Fraction vs Distance from Center (Multiple Timesteps)',
                 fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    plt.tight_layout()
    _save_or_show("shell_hyphal_density_radial.png")


def plot_nutrient_depletion_analysis(nutrient_data_list, timesteps_list, nutrient_name):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(nutrient_data_list)))
    inner_cutoff = CONIDIA_SPAWN_BOX_EXTENT_UM / 2 * np.sqrt(3)

    for i, (nutrient_data, step) in enumerate(zip(nutrient_data_list, timesteps_list)):
        time_hr = step * TIME_STEP_HR
        _radial_curve_with_inner_avg(
            ax, nutrient_data['shell_mid'], nutrient_data['nutrient_delta_avg'],
            colors[i], f'Step {step} ({time_hr:.1f} hr)', inner_cutoff,
        )

    ax.set_xlabel('Distance from center (µm)', fontsize=18)
    ax.set_ylabel(f'{nutrient_name} depletion (mM)', fontsize=18)
    ax.set_title(f'{nutrient_name} Depletion vs Distance from Center (Multiple Timesteps)',
                 fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    plt.tight_layout()
    _save_or_show(f"shell_nutrient_depletion_{_slug(nutrient_name)}.png")


def plot_citric_acid_production_analysis(citric_data_list, timesteps_list):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(citric_data_list)))

    for i, (citric_data, step) in enumerate(zip(citric_data_list, timesteps_list)):
        distances = np.array(citric_data['shell_mid'])
        production = np.array(citric_data['citric_production_mmol'])
        time_hr = step * TIME_STEP_HR
        ax.plot(distances, production, 'o-', linewidth=3, markersize=7,
                color=colors[i], label=f'Step {step} ({time_hr:.1f} hr)')

    ax.set_xlabel('Distance from center (µm)', fontsize=18)
    ax.set_ylabel('Citric acid production (mmol)', fontsize=18)
    ax.set_title('Citric Acid Production vs Distance from Center (Multiple Timesteps)',
                 fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16)
    plt.tight_layout()
    _save_or_show("shell_citric_acid_radial.png")


def export_shell_data_csv(shell_data):
    rows = []
    for i in range(len(shell_data['shell_mid'])):
        rows.append({
            'shell_index':       i,
            'r_inner_um':        float(shell_data['shell_inner'][i]),
            'r_outer_um':        float(shell_data['shell_outer'][i]),
            'r_mid_um':          float(shell_data['shell_mid'][i]),
            'shell_volume_um3':  float(shell_data['shell_volumes'][i]),
            'hyphal_length_um':  float(shell_data['hyphal_lengths'][i]),
            'hyphal_volume_um3': float(shell_data['hyphal_volumes'][i]),
            'hyphal_fraction':   float(shell_data['hyphal_fractions'][i]),
        })
    write_csv(
        'shell_density_analysis.csv',
        fieldnames=['shell_index','r_inner_um','r_outer_um','r_mid_um',
                    'shell_volume_um3','hyphal_length_um','hyphal_volume_um3','hyphal_fraction'],
        rows=rows,
    )


def export_hyphal_density_multistep_csv(shell_data_list, timesteps_list):
    rows = []
    for shell_data, step in zip(shell_data_list, timesteps_list):
        time_hr = float(step * TIME_STEP_HR)
        for i in range(len(shell_data['shell_mid'])):
            rows.append({
                'step':              int(step),
                'time_hr':           time_hr,
                'shell_index':       i,
                'r_inner_um':        float(shell_data['shell_inner'][i]),
                'r_outer_um':        float(shell_data['shell_outer'][i]),
                'r_mid_um':          float(shell_data['shell_mid'][i]),
                'shell_volume_um3':  float(shell_data['shell_volumes'][i]),
                'hyphal_length_um':  float(shell_data['hyphal_lengths'][i]),
                'hyphal_volume_um3': float(shell_data['hyphal_volumes'][i]),
                'hyphal_fraction':   float(shell_data['hyphal_fractions'][i]),
            })
    write_csv(
        'shell_hyphal_density_multistep.csv',
        fieldnames=['step','time_hr','shell_index','r_inner_um','r_outer_um',
                    'r_mid_um','shell_volume_um3','hyphal_length_um',
                    'hyphal_volume_um3','hyphal_fraction'],
        rows=rows,
    )


def export_nutrient_depletion_multistep_csv(nutrient_data_list, timesteps_list, nutrient_name):
    rows = []
    for nutrient_data, step in zip(nutrient_data_list, timesteps_list):
        time_hr = float(step * TIME_STEP_HR)
        for i in range(len(nutrient_data['shell_mid'])):
            rows.append({
                'step':                int(step),
                'time_hr':             time_hr,
                'shell_index':         i,
                'r_inner_um':          float(nutrient_data['shell_inner'][i]),
                'r_outer_um':          float(nutrient_data['shell_outer'][i]),
                'r_mid_um':            float(nutrient_data['shell_mid'][i]),
                'nutrient_before_mM':  float(nutrient_data['nutrient_before_avg'][i]),
                'nutrient_after_mM':   float(nutrient_data['nutrient_after_avg'][i]),
                'nutrient_delta_mM':   float(nutrient_data['nutrient_delta_avg'][i]),
            })
    write_csv(
        f"shell_nutrient_depletion_{_slug(nutrient_name)}.csv",
        fieldnames=['step','time_hr','shell_index','r_inner_um','r_outer_um',
                    'r_mid_um','nutrient_before_mM','nutrient_after_mM','nutrient_delta_mM'],
        rows=rows,
    )


def export_citric_production_multistep_csv(citric_data_list, timesteps_list):
    rows = []
    for citric_data, step in zip(citric_data_list, timesteps_list):
        time_hr = float(step * TIME_STEP_HR)
        for i in range(len(citric_data['shell_mid'])):
            rows.append({
                'step':                    int(step),
                'time_hr':                 time_hr,
                'shell_index':             i,
                'r_inner_um':              float(citric_data['shell_inner'][i]),
                'r_outer_um':              float(citric_data['shell_outer'][i]),
                'r_mid_um':                float(citric_data['shell_mid'][i]),
                'citric_production_mmol':  float(citric_data['citric_production_mmol'][i]),
            })
    write_csv(
        'shell_citric_production_multistep.csv',
        fieldnames=['step','time_hr','shell_index','r_inner_um','r_outer_um',
                    'r_mid_um','citric_production_mmol'],
        rows=rows,
    )


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

    # ---- Pre-loop bookkeeping ----
    history = {
        "total_length_um":       [],
        "total_volume_um3":      [],
        "avg_length_um":         [],
        "avg_volume_um3":        [],
        "avg_tips_per_hypha":    [],
        "n_hyphae":              [],
        "total_branch_nodes":    [],
        "total_tips":            [],
        "citric_total_mmol":     [],
        "citric_mM":             [],
        "liquid_volume_L":       [],
        "mean_nutrients":        {k: [] for k in NUTRIENT_SPECS.keys()},
        "apical_branch_events":  [],
        "lateral_branch_events": [],
    }

    branch_event_counters = {"apical": 0, "lateral": 0}

    # Snapshot timesteps: 20%, 40%, 60%, 80%, 100% of N_STEPS (clamped to N_STEPS-1).
    snapshot_timesteps = sorted(set(
        min(int(N_STEPS * i / 5), N_STEPS - 1) for i in range(1, 6)
    ))
    graph_snapshots    = {}  # {timestep: deep_copy_of_graphs}
    nutrient_snapshots = {}  # {timestep: {nutrient_name: {'before': arr, 'after': arr}}}
    citric_snapshots   = {}  # {timestep: {'before': arr, 'after': arr}}

    # ---- Multiprocessing setup (consume pass; skipped if N_WORKERS == 1) ----
    use_parallel = N_WORKERS > 1
    pool = None
    shared_state = None

    if use_parallel:
        import atexit

        def _cleanup_parallel():
            global pool, shared_state
            if pool is not None:
                try:
                    pool.close()
                    pool.join()
                except Exception:
                    pass
                pool = None
            if shared_state is not None:
                shared_state.cleanup()
                shared_state = None

        print(f"[parallel] Spawning {N_WORKERS} worker processes for the consume pass...")
        shared_state = _SharedGridState(grid)
        pool = mp.Pool(
            processes=N_WORKERS,
            initializer=_consume_worker_init,
            initargs=(shared_state.metadata,),
        )
        atexit.register(_cleanup_parallel)
        print(f"[parallel] Nutrient averaging will use up to {min(N_WORKERS, len(NUTRIENT_SPECS))} threads.")
    else:
        print("[parallel] N_WORKERS=1 — running consume pass serially.")

    # ---- Main loop: branching → consumption → metrics → averaging ----
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

        # 3a. Snapshot nutrient + citric arrays BEFORE consumption (for radial analysis)
        if step in snapshot_timesteps:
            nutrient_snapshots[step] = {
                nutr: {"before": copy.deepcopy(grid.nutrients[nutr])}
                for nutr in ("Nitrogen", "Phosphorus", "Oxygen", "Glucose")
            }
            citric_snapshots[step] = {
                "before": copy.deepcopy(grid.metabolites["citric_acid_spatial_mM"])
            }

        # 3. Consumption pass: N/P/O + Glucose; CA produced as a by-product
        if use_parallel:
            total_ca_mmol_step, n_ca_events = consume_nutrients_parallel(
                grid, graphs, pool, shared_state
            )
        else:
            total_ca_mmol_step, n_ca_events = consume_nutrients(grid, graphs)

        # 3b. Snapshot the same arrays AFTER consumption
        if step in snapshot_timesteps:
            for nutr in ("Nitrogen", "Phosphorus", "Oxygen", "Glucose"):
                nutrient_snapshots[step][nutr]["after"] = copy.deepcopy(grid.nutrients[nutr])
            citric_snapshots[step]["after"] = copy.deepcopy(
                grid.metabolites["citric_acid_spatial_mM"]
            )

        # 4. Record metrics (post-consumption, pre-averaging)
        m = compute_metrics(grid, graphs)
        history["total_length_um"]      .append(m["total_length_um"])
        history["total_volume_um3"]     .append(m["total_volume_um3"])
        history["avg_length_um"]        .append(m["avg_length_um"])
        history["avg_volume_um3"]       .append(m["avg_volume_um3"])
        history["avg_tips_per_hypha"]   .append(m["avg_tips_per_hypha"])
        history["n_hyphae"]             .append(m["n_hyphae"])
        history["total_branch_nodes"]   .append(m["total_branch_nodes"])
        history["total_tips"]           .append(m["total_tips"])
        history["citric_total_mmol"]    .append(m["citric_total_mmol"])
        history["citric_mM"]            .append(m["citric_mM"])
        history["liquid_volume_L"]      .append(m["liquid_volume_L"])
        history["apical_branch_events"] .append(branch_event_counters["apical"])
        history["lateral_branch_events"].append(branch_event_counters["lateral"])

        nut_means = mean_nutrients_over_liquid(grid)
        for nutr in NUTRIENT_SPECS.keys():
            history["mean_nutrients"][nutr].append(nut_means.get(nutr, float("nan")))

        # 5. Nutrient averaging (well-mixed bath re-equilibration).
        #    Parallel path: 4 species resampled simultaneously via ThreadPoolExecutor.
        #    Serial path: species resampled sequentially using the module-level RNG.
        if use_parallel:
            average_nutrients_parallel(grid, step)
        else:
            average_nutrients(grid)

        # 6. Save post-step graph state for radial-density analysis
        if step in snapshot_timesteps:
            graph_snapshots[step] = copy.deepcopy(graphs)

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

    # ----- Shell-based radial analyses -----
    print("\n=== Performing hyphal density analysis ===")
    pellet_center = (
        np.mean(np.asarray(conidia, dtype=float), axis=0)
        if len(conidia) > 0 else grid.origin_um
    )

    # Compute shell radii once from FINAL graphs so every snapshot uses identical bins.
    _, shell_radii = compute_shell_boundaries(pellet_center, graphs, N_SHELLS)

    timesteps_list = sorted(graph_snapshots.keys())
    shell_data_list = []
    for step in timesteps_list:
        print(f"  Analyzing hyphal density at step {step}...")
        shell_data_list.append(
            analyze_hyphal_density_by_shell(pellet_center, graph_snapshots[step], n_shells=N_SHELLS)
        )

    if shell_data_list:
        export_shell_data_csv(shell_data_list[-1])
        export_hyphal_density_multistep_csv(shell_data_list, timesteps_list)

        print("Plotting cross-section with all shells (final timestep)...")
        plot_shell_cross_section(pellet_center, graphs, shell_data_list[-1])

        middle_shell_idx = len(shell_data_list[-1]['shell_mid']) // 2
        print(f"Plotting cross-section with shell {middle_shell_idx} highlighted...")
        plot_shell_cross_section(pellet_center, graphs, shell_data_list[-1],
                                 highlight_shell_idx=middle_shell_idx)

        print("Plotting hyphal fraction for multiple timesteps...")
        plot_hyphal_density_analysis(shell_data_list, timesteps_list)

    print("\n=== Performing nutrient depletion analysis ===")
    for nutrient_name in ("Nitrogen", "Phosphorus", "Oxygen", "Glucose"):
        print(f"  Analyzing {nutrient_name} depletion...")
        nutrient_data_list = []
        for step in timesteps_list:
            before = nutrient_snapshots[step][nutrient_name]['before']
            after  = nutrient_snapshots[step][nutrient_name]['after']
            nutrient_data_list.append(
                analyze_nutrient_by_shell(pellet_center, grid, before, after,
                                          nutrient_name, shell_radii)
            )
        if nutrient_data_list:
            export_nutrient_depletion_multistep_csv(nutrient_data_list, timesteps_list, nutrient_name)
            print(f"  Plotting {nutrient_name} depletion for multiple timesteps...")
            plot_nutrient_depletion_analysis(nutrient_data_list, timesteps_list, nutrient_name)

    print("\n=== Performing citric acid production analysis ===")
    citric_data_list = []
    for step in timesteps_list:
        print(f"  Analyzing citric acid production at step {step}...")
        before = citric_snapshots[step]['before']
        after  = citric_snapshots[step]['after']
        citric_data_list.append(
            analyze_citric_acid_by_shell(pellet_center, grid, before, after, shell_radii)
        )
    if citric_data_list:
        export_citric_production_multistep_csv(citric_data_list, timesteps_list)
        print("Plotting citric acid production for multiple timesteps...")
        plot_citric_acid_production_analysis(citric_data_list, timesteps_list)
