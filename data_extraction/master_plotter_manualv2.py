import os
import re
import csv
import time
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
from mpl_toolkits.mplot3d import proj3d

# Optional (for convex hull volume). If not available, volume plot (8) will degrade gracefully.
try:
    from scipy.spatial import ConvexHull
    _HAS_HULL = True
except Exception:
    _HAS_HULL = False

# Use a faster backend for better interactivity
try:
    import matplotlib
    matplotlib.use("TkAgg")
except:
    pass

# =============================================================================
# Tunables
# =============================================================================

DATASET_DIR = r"C:\Users\asj21\OneDrive - Imperial College London\Desktop\Avyay Jamadagni PhD\Projects\Mycelium growth sim\3D_image_grapher\Mndataset"

BRIGHTNESS_THRESH = 30

# show/select on a global downsample of ~8000 points
DISPLAY_DOWNSAMPLE_N = 9000
DISPLAY_DOWNSAMPLE_SEED = 123

# Filtering (optional; keeps your original concept but cheaper than O(N^2))
FILTER_ENABLE = True
FILTER_NEIGHBOR_RADIUS = 10.0
FILTER_MIN_NEIGHBORS = 3
FILTER_KNN = 20  # uses kNN then counts within radius

# Skeleton params
CONTRACTION_K = 45
CONTRACTION_ITERS = 15
WL0 = 1.6
WH0 = 1.2
STOP_RATIO = 0.02

SAMPLING_KDIR = 16
L_THRESH = 0.85
R_LINE = 3.5
R_BRANCH = 1.8

CONNECT_K = 3          # initial kNN connectivity for skeleton keypoints
FORCE_CONNECT_MST = True  # "connect everything no matter how far away it is!"

CALIB_RADIUS = 6.0
CALIB_ITERS = 2

ZLIM = (-200, 200)

# =============================================================================
# Reporting/export Tunables
# =============================================================================

EXPORT_ROOT_NAME = "skeleton_metrics_export"
EXPORT_POINTS_SUBDIR = "hypha_points"
EXPORT_SKEL_SUBDIR = "hypha_skeleton"

# spur rule: if branch node (deg>=3) has a neighbor whose branch chain length is 1 or 2 nodes -> those 1 or 2 nodes are classified as spur
SPUR_MAX_NODES = 2

# Varying radius along skeleton:
# - Assign each original point to its nearest skeleton node -> distances define local radius at node
# - For each edge, use average radius of endpoints
# Optional smoothing over graph to reduce noise
RADIUS_SMOOTH_ITERS = 1  # 0 = no smoothing, 1-3 usually fine

# =============================================================================
# Utilities
# =============================================================================

def extract_slice_number(filename: str) -> int:
    m = re.search(r"([-+]\d{3})\.png$", filename)
    return int(m.group(1)) if m else 0

def _cmap_for_n(n: int):
    return plt.cm.get_cmap("tab20" if n <= 20 else "hsv")

def _color_for_id(hid: int, n_ids: int, cmap):
    if n_ids <= 1:
        return cmap(0.0)
    return cmap((hid % 20) / 19.0)

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def _timestamp_tag():
    return time.strftime("%Y%m%d_%H%M%S")

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# =============================================================================
# Manual picker (box select, no grow tool)
# =============================================================================

class ManualPointPicker3D:
    def __init__(self, points_xyz: np.ndarray, title="Manual picker", points_s=3, bg_alpha=0.05):
        self.P = points_xyz.astype(np.float64)
        self.n = len(self.P)

        self.assign = np.full(self.n, -1, dtype=int)
        self.active = 0

        self.mode = "select"
        self.accepted = False
        self.quit_all = False
        self._shift_down = False

        self.points_s = points_s
        self.bg_alpha = bg_alpha
        self.title = title

        self.fig = None
        self.ax = None
        self.scat = None

        # selection mode
        self.selector_mode = "rect"   # "rect" or "lasso"
        self.lasso = None

        # custom box-drag state (pixels)
        self._dragging = False
        self._drag_start = None  # (x,y) in pixels

    def _colors_for_points(self):
        cmap = plt.cm.get_cmap("tab20")
        cols = np.zeros((self.n, 4), dtype=float)

        un = (self.assign == -1)
        cols[un] = np.array([0, 0, 0, self.bg_alpha])

        assigned = ~un
        if np.any(assigned):
            ids = self.assign[assigned]
            t = (ids % 20) / 19.0
            cols[assigned] = cmap(t)
            cols[assigned, 3] = 0.95
        return cols

    def _apply_scatter_colors(self, cols_rgba):
        self.scat.set_facecolors(cols_rgba)
        self.scat.set_edgecolors(cols_rgba)
        if hasattr(self.scat, "_facecolor3d"):
            self.scat._facecolor3d = cols_rgba
        if hasattr(self.scat, "_edgecolor3d"):
            self.scat._edgecolor3d = cols_rgba

    def _update_plot(self):
        self._apply_scatter_colors(self._colors_for_points())
        assigned_count = int(np.sum(self.assign != -1))
        sel_name = "RECT" if self.selector_mode == "rect" else "LASSO"
        self.ax.set_title(
            f"{self.title}\n"
            f"Mode={self.mode.upper()} | Selector={sel_name} | Active={self.active} | Assigned={assigned_count}/{self.n}\n"
            f"MMB drag=ROTATE | LMB drag=SELECT | SHIFT+select=ERASE | ←/→=CHANGE ID (0-20) | l=toggle | Enter=review | a=accept | q=quit"
        )
        self.fig.canvas.draw_idle()

    def _project_to_pixels(self):
        x, y, z = self.P[:, 0], self.P[:, 1], self.P[:, 2]
        x2, y2, _ = proj3d.proj_transform(x, y, z, self.ax.get_proj())
        return self.ax.transData.transform(np.column_stack([x2, y2]))  # pixels

    def _assign_indices(self, idx):
        if idx.size == 0:
            print("Selection hit 0 points.")
            return
        if self._shift_down:
            self.assign[idx] = -1
            print(f"Erased {idx.size} points.")
        else:
            self.assign[idx] = self.active
            print(f"Assigned {idx.size} points to hypha {self.active}.")
        self._update_plot()

    def _on_mouse_press(self, event):
        if self.mode != "select":
            return
        if self.selector_mode != "rect":
            return
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.x is None or event.y is None:
            return
        self._dragging = True
        self._drag_start = (float(event.x), float(event.y))

    def _on_mouse_release(self, event):
        if not self._dragging:
            return
        self._dragging = False

        if event.x is None or event.y is None or self._drag_start is None:
            self._drag_start = None
            return

        x0, y0 = self._drag_start
        x1, y1 = float(event.x), float(event.y)
        self._drag_start = None

        xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
        ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)

        if (xmax - xmin) < 2 or (ymax - ymin) < 2:
            return

        xy = self._project_to_pixels()
        inside = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
        idx = np.where(inside)[0]
        self._assign_indices(idx)

    def _on_select_lasso(self, verts):
        if self.mode != "select":
            return
        if verts is None or len(verts) < 3:
            return
        poly = Path(verts)  # pixels
        xy = self._project_to_pixels()
        inside = poly.contains_points(xy)
        idx = np.where(inside)[0]
        self._assign_indices(idx)

    def _enable_lasso(self):
        if self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None
        self.lasso = LassoSelector(self.ax, onselect=self._on_select_lasso, button=1)

    def _disable_lasso(self):
        if self.lasso is not None:
            self.lasso.disconnect_events()
            self.lasso = None

    def _set_selector_mode(self, mode):
        self.selector_mode = mode
        if mode == "lasso":
            self._enable_lasso()
        else:
            self._disable_lasso()
        self._update_plot()

    def _on_key_press(self, event):
        if event.key is None:
            return
        if event.key == "shift":
            self._shift_down = True
            return

        if event.key == "right" and self.mode == "select":
            self.active = min(self.active + 1, 20)
            self._update_plot()
            return

        if event.key == "left" and self.mode == "select":
            self.active = max(self.active - 1, 0)
            self._update_plot()
            return

        if event.key == "l" and self.mode == "select":
            self._set_selector_mode("lasso" if self.selector_mode == "rect" else "rect")
            return

        if event.key == "enter":
            self.mode = "review" if self.mode == "select" else "select"
            self._update_plot()
            return

        if event.key == "a" and self.mode == "review":
            self.accepted = True
            plt.close(self.fig)
            return

        if event.key == "q":
            self.quit_all = True
            plt.close(self.fig)
            return

        if event.key == "x" and self.mode == "select":
            self.assign[:] = -1
            self._update_plot()
            return

        if event.key == "c" and self.mode == "select":
            self.assign[self.assign == self.active] = -1
            self._update_plot()
            return

    def _on_key_release(self, event):
        if event.key == "shift":
            self._shift_down = False

    def run(self):
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")

        self.ax.mouse_init(rotate_btn=2, zoom_btn=3, pan_btn=99)

        self.scat = self.ax.scatter(
            self.P[:, 0], self.P[:, 1], self.P[:, 2],
            s=self.points_s,
            depthshade=False
        )
        self.ax.set_zlim(ZLIM[0], ZLIM[1])
        self.ax.view_init(elev=20, azim=45)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("key_release_event", self._on_key_release)

        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        self.fig.canvas.draw()
        self._set_selector_mode("rect")

        plt.tight_layout()
        plt.show()
        return self.assign, self.accepted, self.quit_all

# =============================================================================
# Skeletonisation pipeline (your methods)
# =============================================================================

def build_knn(points: np.ndarray, k: int):
    n = len(points)
    if n == 0:
        return np.empty((0, 0), dtype=int), np.empty((0, 0), dtype=float)
    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(points)
    dists, idxs = nn.kneighbors(points)
    return idxs[:, 1:], dists[:, 1:]

def graph_laplacian_from_knn(idxs: np.ndarray, dists: np.ndarray, sigma: float | None = None) -> csr_matrix:
    n = idxs.shape[0]
    if n == 0:
        return csr_matrix((0, 0))
    if sigma is None:
        sigma = float(np.median(dists)) + 1e-6

    rows, cols, data = [], [], []
    for i in range(n):
        for j, d in zip(idxs[i], dists[i]):
            w = np.exp(-(d * d) / (2.0 * sigma * sigma))
            rows.append(i); cols.append(int(j)); data.append(w)

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = 0.5 * (W + W.T)
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = diags(deg)
    return D - W

def laplacian_contraction(points: np.ndarray,
                          k: int = 12,
                          max_iter: int = 15,
                          wl0: float = 2.0,
                          wh0: float = 1.0,
                          stop_ratio: float = 0.01) -> np.ndarray:
    P = points.astype(np.float64).copy()
    if len(P) < 10:
        return P

    idxs0, d0 = build_knn(P, k)
    S0 = d0.mean(axis=1) + 1e-6

    wl = wl0
    wh = wh0

    for _ in range(max_iter):
        idxs, dists = build_knn(P, k)
        St = dists.mean(axis=1) + 1e-6

        L = graph_laplacian_from_knn(idxs, dists)

        WH = wh * (S0 / St)
        A = (wl * L) + diags(WH)

        Bx = WH * P[:, 0]
        By = WH * P[:, 1]
        Bz = WH * P[:, 2]

        Px = spsolve(A.tocsr(), Bx)
        Py = spsolve(A.tocsr(), By)
        Pz = spsolve(A.tocsr(), Bz)

        P_new = np.column_stack([Px, Py, Pz])

        shrink_ratio = float(np.median(St / S0))
        wl_new = wl * shrink_ratio

        P = P_new
        if wl > 0 and (wl_new / wl) < stop_ratio:
            break
        wl = wl_new

    return P

def directionality_degree(points: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    n = len(points)
    l = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nb = points[idxs[i]]
        c = nb.mean(axis=0)
        X = nb - c
        C = X.T @ X
        evals = np.linalg.eigvalsh(C)
        s = float(evals.sum()) + 1e-12
        l[i] = float(evals[-1]) / s
    return l

def adaptive_sampling(points: np.ndarray,
                      k_dir: int = 16,
                      l_thresh: float = 0.90,
                      r_line: float = 6.0,
                      r_branch: float = 2.5):
    if len(points) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    idxs, _ = build_knn(points, k_dir)
    lvals = directionality_degree(points, idxs)

    radii = np.where(lvals >= l_thresh, r_line, r_branch)
    order = np.argsort(lvals)  # junction-ish first

    selected = []
    selected_pts = np.empty((0, 3), dtype=np.float64)

    for idx in order:
        p = points[idx:idx+1]
        r = float(radii[idx])

        if len(selected) == 0:
            selected.append(int(idx))
            selected_pts = p.copy()
            continue

        dmin = np.linalg.norm(selected_pts - p, axis=1).min()
        if dmin >= r:
            selected.append(int(idx))
            selected_pts = np.vstack([selected_pts, p])

    return np.array(selected, dtype=int), lvals

def connect_skeleton_points(points: np.ndarray, k: int = 3):
    n = len(points)
    if n < 2:
        return []

    k_eff = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(points)
    _, idxs = nn.kneighbors(points)

    edges = set()
    for i in range(n):
        for j in idxs[i, 1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            edges.add((a, b))
    return sorted(edges)

def mst_edges(points: np.ndarray):
    """Always connect everything with MST."""
    if len(points) < 2:
        return []
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    k = min(12, len(points)-1)
    nn = NearestNeighbors(n_neighbors=k+1).fit(points)
    dists, idxs = nn.kneighbors(points)
    for i in range(len(points)):
        for j, d in zip(idxs[i, 1:], dists[i, 1:]):
            G.add_edge(i, int(j), weight=float(d))
    T = nx.minimum_spanning_tree(G, weight="weight")
    return sorted((min(u, v), max(u, v)) for u, v in T.edges())

def remove_loops_via_mst(points: np.ndarray, edges):
    """Replace graph with MST over given edges (loop removal)."""
    if len(points) < 2:
        return []
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    for i, j in edges:
        w = float(np.linalg.norm(points[i] - points[j]))
        G.add_edge(i, j, weight=w)
    T = nx.minimum_spanning_tree(G, weight="weight")
    return sorted((min(u, v), max(u, v)) for u, v in T.edges())

def calibrate_skeleton(nodes: np.ndarray, edges, original_cluster_points: np.ndarray,
                       radius: float = 6.0, iters: int = 2):
    if len(nodes) == 0 or len(original_cluster_points) == 0:
        return nodes

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)

    nn = NearestNeighbors(n_neighbors=min(300, len(original_cluster_points))).fit(original_cluster_points)
    nodes_new = nodes.astype(np.float64).copy()

    for _ in range(iters):
        updated = nodes_new.copy()
        for i in range(len(nodes_new)):
            nbrs = list(G.neighbors(i))
            if len(nbrs) == 0:
                continue

            pi = nodes_new[i]

            v = np.zeros(3, dtype=np.float64)
            for j in nbrs:
                d = nodes_new[j] - pi
                nrm = np.linalg.norm(d) + 1e-12
                v += d / nrm
            vnorm = np.linalg.norm(v)
            if vnorm < 1e-9:
                continue
            t = v / vnorm

            dists, idxs = nn.kneighbors(pi.reshape(1, 3), return_distance=True)
            dists = dists.ravel()
            idxs = idxs.ravel()
            mask = dists <= radius
            if not np.any(mask):
                continue

            P = original_cluster_points[idxs[mask]]
            V = P - pi
            proj = P - (V @ t)[:, None] * t[None, :]
            updated[i] = proj.mean(axis=0)

        nodes_new = updated

    return nodes_new

def skeletonise_cluster(points_xyz: np.ndarray):
    if len(points_xyz) < 10:
        return {"nodes": np.empty((0, 3)), "edges": [], "keypoints": np.empty((0, 3)), "contracted": points_xyz}

    contracted = laplacian_contraction(points_xyz, k=CONTRACTION_K, max_iter=CONTRACTION_ITERS,
                                       wl0=WL0, wh0=WH0, stop_ratio=STOP_RATIO)

    sel_idx, _ = adaptive_sampling(contracted, k_dir=SAMPLING_KDIR, l_thresh=L_THRESH,
                                   r_line=R_LINE, r_branch=R_BRANCH)
    keypoints = contracted[sel_idx] if len(sel_idx) else np.empty((0, 3))

    if len(keypoints) < 2:
        return {"contracted": contracted, "keypoints": keypoints, "nodes": keypoints, "edges": []}

    edges = connect_skeleton_points(keypoints, k=CONNECT_K)
    edges = remove_loops_via_mst(keypoints, edges)

    if FORCE_CONNECT_MST:
        edges = mst_edges(keypoints)

    nodes = calibrate_skeleton(keypoints, edges, original_cluster_points=points_xyz,
                               radius=CALIB_RADIUS, iters=CALIB_ITERS)

    return {"contracted": contracted, "keypoints": keypoints, "nodes": nodes, "edges": edges}

# =============================================================================
# Filtering/downsampling helpers
# =============================================================================

def downsample_points(points_xyz: np.ndarray, target_n: int, seed: int = 0) -> np.ndarray:
    if len(points_xyz) <= target_n:
        return points_xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points_xyz), size=target_n, replace=False)
    return points_xyz[idx]

def filter_points_knn_radius(points_xyz: np.ndarray, radius: float, min_neighbors: int, k: int) -> np.ndarray:
    if len(points_xyz) < max(10, min_neighbors + 1):
        return points_xyz
    k_eff = min(k, len(points_xyz) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(points_xyz)
    dists, _ = nn.kneighbors(points_xyz)
    within = (dists[:, 1:] <= radius)
    counts = within.sum(axis=1)
    keep = counts >= min_neighbors
    return points_xyz[keep]

# =============================================================================
# Plot skeleton overlay after acceptance
# =============================================================================

def plot_skeletonised_overlay(frame_name: str,
                              global_points_xyz: np.ndarray,
                              assign: np.ndarray,
                              skels_by_hypha: dict[int, dict],
                              title_suffix="",
                              points_alpha=0.04,
                              points_s=2,
                              node_s=24,
                              node_alpha=0.95,
                              edge_lw=1.5,
                              edge_alpha=0.9):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.mouse_init(rotate_btn=2, zoom_btn=3)

    cmap = _cmap_for_n(20)

    un = (assign == -1)
    if np.any(un):
        pts = global_points_xyz[un]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="black", s=points_s, alpha=points_alpha)

    for hid in sorted(set(assign) - {-1}):
        m = (assign == hid)
        pts = global_points_xyz[m]
        color = _color_for_id(hid, 20, cmap)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=[color], s=points_s, alpha=points_alpha * 2.0)

    for hid, sk in skels_by_hypha.items():
        nodes = sk.get("nodes", None)
        edges = sk.get("edges", [])
        if nodes is None or len(nodes) == 0:
            continue
        color = _color_for_id(hid, 20, cmap)
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c=[color], s=node_s, alpha=node_alpha)
        for (i, j) in edges:
            p, q = nodes[i], nodes[j]
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=edge_lw, alpha=edge_alpha)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlim(ZLIM[0], ZLIM[1])
    ax.set_title(f"{frame_name}: skeletonised manual hyphae {title_suffix}".strip())
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()

# =============================================================================
# Load frame points
# =============================================================================

def load_frame_points(frame_dir: str) -> np.ndarray:
    image_files = [f for f in os.listdir(frame_dir) if f.endswith(".png")]
    if not image_files:
        return np.empty((0, 3), dtype=np.float64)

    sorted_files = sorted(image_files, key=extract_slice_number)

    all_coords = []
    for fn in sorted_files:
        fp = os.path.join(frame_dir, fn)
        z = extract_slice_number(fn)

        img = Image.open(fp)
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        rgb = np.array(img)[:, :, :3]
        brightness = np.sum(rgb, axis=2)
        mask = brightness > BRIGHTNESS_THRESH
        y, x = np.where(mask)
        if len(x) > 0:
            coords = np.column_stack([x, y, np.full_like(x, z)])
            all_coords.append(coords)

    if not all_coords:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(all_coords).astype(np.float64)

# =============================================================================
# Pixel->micrometer calibration (click 2 points in XY of first frame)
# =============================================================================

def calibrate_um_per_pixel(first_frame_points_xyz: np.ndarray) -> float:
    if len(first_frame_points_xyz) < 2:
        print("Not enough points for calibration; defaulting um_per_pixel=1.0")
        return 1.0

    pts = first_frame_points_xyz
    if len(pts) > 50000:
        pts = downsample_points(pts, 50000, seed=1234)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.15)
    ax.set_title(
        "Calibration (Frame 1, top-down XY)\n"
        "Click TWO points, then close window if it doesn't auto-close."
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    clicked = plt.ginput(2, timeout=0)
    plt.close(fig)

    if clicked is None or len(clicked) < 2:
        print("Calibration cancelled; defaulting um_per_pixel=1.0")
        return 1.0

    (x0, y0), (x1, y1) = clicked[0], clicked[1]
    dist_pix = float(np.hypot(x1 - x0, y1 - y0))
    if dist_pix < 1e-9:
        print("Clicked points too close; defaulting um_per_pixel=1.0")
        return 1.0

    while True:
        try:
            dist_um = float(input(f"Pixel distance = {dist_pix:.3f} px. Enter the real distance in micrometers (µm): ").strip())
            if dist_um <= 0:
                raise ValueError("distance must be > 0")
            break
        except Exception as e:
            print(f"Invalid input: {e}. Try again.")

    um_per_pix = dist_um / dist_pix
    print(f"Calibration set: {um_per_pix:.6f} µm/pixel")
    return um_per_pix

# =============================================================================
# Node classification + spurs + metrics
# =============================================================================

def build_graph_from_skeleton(nodes: np.ndarray, edges):
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)
    return G

def _walk_branch_chain(G: nx.Graph, start_from_branch: int, neighbor: int, max_nodes: int = 100000):
    chain = []
    prev = start_from_branch
    curr = neighbor
    steps = 0
    while True:
        chain.append(curr)
        steps += 1
        if steps > max_nodes:
            break
        deg = G.degree(curr)
        if deg != 2:
            break
        nbrs = list(G.neighbors(curr))
        nxt = nbrs[0] if nbrs[1] == prev else nbrs[1]
        prev, curr = curr, nxt
    return chain

def classify_spurs(nodes: np.ndarray, edges, spur_max_nodes: int = 2):
    if len(nodes) == 0:
        return set(), 0

    G = build_graph_from_skeleton(nodes, edges)
    spur_nodes = set()
    spur_count = 0

    for b in G.nodes:
        if G.degree(b) >= 3:
            for nb in G.neighbors(b):
                chain = _walk_branch_chain(G, b, nb)
                if 1 <= len(chain) <= spur_max_nodes:
                    for u in chain:
                        spur_nodes.add(u)
                    spur_count += 1

    return spur_nodes, spur_count

def skeleton_length_um(nodes: np.ndarray, edges, um_per_pix: float, exclude_nodes: set[int] | None = None) -> float:
    if len(nodes) < 2 or len(edges) == 0:
        return 0.0
    exclude_nodes = exclude_nodes or set()
    total = 0.0
    for i, j in edges:
        if i in exclude_nodes or j in exclude_nodes:
            continue
        total += float(np.linalg.norm(nodes[i] - nodes[j])) * um_per_pix
    return total

def count_tips_excluding_spurs(nodes: np.ndarray, edges, spur_nodes: set[int]) -> int:
    if len(nodes) == 0:
        return 0
    G = build_graph_from_skeleton(nodes, edges)
    tips = [n for n in G.nodes if G.degree(n) == 1 and n not in spur_nodes]
    return int(len(tips))

def count_branch_nodes_with_proper_branches(nodes: np.ndarray, edges, spur_max_nodes: int) -> int:
    if len(nodes) == 0:
        return 0
    G = build_graph_from_skeleton(nodes, edges)
    cnt = 0
    for b in G.nodes:
        if G.degree(b) >= 3:
            has_proper = False
            for nb in G.neighbors(b):
                chain = _walk_branch_chain(G, b, nb)
                if len(chain) > spur_max_nodes:
                    has_proper = True
                    break
            if has_proper:
                cnt += 1
    return int(cnt)

# =============================================================================
# NEW: radius varying along skeleton + volume
# =============================================================================

def estimate_node_radii_um(points_xyz: np.ndarray, nodes: np.ndarray, um_per_pix: float, smooth_iters: int = 0):
    """
    Local radius at each skeleton node:
      - assign each original point to nearest skeleton node
      - node radius = median distance of its assigned points to that node
      - fill missing nodes with global median
      - optionally smooth radii along the skeleton graph (degree-weighted neighbor averaging)
    Returns radii_um array length Nnodes.
    """
    n_nodes = len(nodes)
    if len(points_xyz) < 5 or n_nodes < 2:
        return np.zeros(n_nodes, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=1).fit(nodes)
    dists, idxs = nn.kneighbors(points_xyz)
    dists = dists.ravel()
    idxs = idxs.ravel().astype(int)

    # collect distances per node
    buckets = [[] for _ in range(n_nodes)]
    for d, i in zip(dists, idxs):
        buckets[i].append(float(d))

    radii_pix = np.full(n_nodes, np.nan, dtype=np.float64)
    for i in range(n_nodes):
        if buckets[i]:
            radii_pix[i] = float(np.median(buckets[i]))

    # fill missing with global median (of available)
    finite = np.isfinite(radii_pix)
    if np.any(finite):
        fill_val = float(np.median(radii_pix[finite]))
        radii_pix[~finite] = fill_val
    else:
        radii_pix[:] = 0.0

    # optional smoothing needs a graph; caller can smooth after excluding spurs if desired
    radii_um = radii_pix * um_per_pix
    if smooth_iters <= 0:
        return radii_um

    # build kNN graph on nodes for smoothing (fast, avoids needing edges here)
    # NOTE: we do NOT change your skeleton edges logic; smoothing is just for radius estimation
    k = min(6, n_nodes - 1)
    nn2 = NearestNeighbors(n_neighbors=k + 1).fit(nodes)
    _, neigh = nn2.kneighbors(nodes)
    neigh = neigh[:, 1:]

    r = radii_um.copy()
    for _ in range(smooth_iters):
        r_new = r.copy()
        for i in range(n_nodes):
            nb = neigh[i]
            if len(nb) == 0:
                continue
            r_new[i] = 0.5 * r[i] + 0.5 * float(np.mean(r[nb]))
        r = r_new

    return r

def hypha_volume_um3_variable_radius(points_xyz: np.ndarray, nodes: np.ndarray, edges, um_per_pix: float,
                                     exclude_nodes: set[int] | None = None, smooth_iters: int = 0) -> float:
    """
    Cylinder-sum volume with varying radius:
      V = sum_edges pi * r_edge^2 * L_edge
      r_edge = (r_i + r_j)/2 using per-node radii.
    Spur edges removed by exclude_nodes.
    """
    exclude_nodes = exclude_nodes or set()
    if len(nodes) < 2 or len(edges) == 0 or len(points_xyz) < 5:
        return 0.0

    r_nodes_um = estimate_node_radii_um(points_xyz, nodes, um_per_pix, smooth_iters=smooth_iters)

    V = 0.0
    for i, j in edges:
        if i in exclude_nodes or j in exclude_nodes:
            continue
        L_um = float(np.linalg.norm(nodes[i] - nodes[j])) * um_per_pix
        r_edge = 0.5 * (float(r_nodes_um[i]) + float(r_nodes_um[j]))
        if L_um > 0 and r_edge > 0:
            V += math.pi * (r_edge ** 2) * L_um
    return float(V)

def global_prism_volume_um3(all_skel_nodes_xyz: np.ndarray, um_per_pix: float, inflate_radius_um: float):
    """
    Prism-like: XY cross-section = convex hull of skeleton XY, inflated by radius, height = Z range.
    """
    if not _HAS_HULL:
        return 0.0
    if len(all_skel_nodes_xyz) < 3:
        return 0.0

    xy = all_skel_nodes_xyz[:, :2].astype(np.float64)
    z = all_skel_nodes_xyz[:, 2].astype(np.float64)
    height_um = (float(z.max() - z.min()) * um_per_pix)
    if height_um <= 0:
        return 0.0

    try:
        hull = ConvexHull(xy)
        verts = xy[hull.vertices]
        area_pix2 = float(hull.volume)  # 2D area
        perim_pix = 0.0
        for i in range(len(verts)):
            p = verts[i]
            q = verts[(i + 1) % len(verts)]
            perim_pix += float(np.linalg.norm(p - q))

        r_pix = float(inflate_radius_um / um_per_pix) if um_per_pix > 0 else 0.0
        area_infl_pix2 = area_pix2 + perim_pix * r_pix + math.pi * (r_pix ** 2)
        area_infl_um2 = area_infl_pix2 * (um_per_pix ** 2)
        return float(area_infl_um2 * height_um)
    except Exception:
        return 0.0

# =============================================================================
# End-of-run plotting
# =============================================================================

def plot_time_series(frames, y, title, ylabel, xlabel="Frame"):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(frames, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

def plot_subplots_per_hypha(frames, hypha_ids, series_dict, title, ylabels, ncols=3):
    n = len(hypha_ids)
    if n == 0:
        return

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))

    fig = plt.figure(figsize=(5 * ncols, 3.8 * nrows))
    fig.suptitle(title)

    for idx, hid in enumerate(hypha_ids):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        for name in ylabels:
            vals = series_dict.get(hid, {}).get(name, [])
            ax.plot(frames, vals, marker="o", label=name)
        ax.set_title(f"Hypha {hid}")
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.25)
        ax.legend()

    plt.tight_layout()
    plt.show()

# =============================================================================
# Main
# =============================================================================

def main():
    frame_folders = [d for d in os.listdir(DATASET_DIR)
                     if os.path.isdir(os.path.join(DATASET_DIR, d)) and d.startswith("frame")]
    frame_folders = sorted(frame_folders, key=lambda s: int(re.search(r"\d+", s).group()))
    print(f"Found {len(frame_folders)} frame folders")
    if len(frame_folders) == 0:
        return

    export_root = os.path.join(DATASET_DIR, f"{EXPORT_ROOT_NAME}_{_timestamp_tag()}")
    points_dir = os.path.join(export_root, EXPORT_POINTS_SUBDIR)
    skel_dir = os.path.join(export_root, EXPORT_SKEL_SUBDIR)
    _safe_mkdir(export_root)
    _safe_mkdir(points_dir)
    _safe_mkdir(skel_dir)
    print(f"Exporting to: {export_root}")

    # calibration before selection
    first_frame_dir = os.path.join(DATASET_DIR, frame_folders[0])
    pts_first_all = load_frame_points(first_frame_dir)
    um_per_pix = calibrate_um_per_pixel(pts_first_all)

    global_rows = []
    hypha_rows = []

    frames_numeric = []
    global_series = {
        "global_nodes": [],
        "global_branch_nodes_proper": [],
        "global_spur_nodes": [],
        "global_quality": [],
        "global_tips": [],
        "global_volume_prism": [],
        "global_volume_sumhypha": [],
    }

    hypha_series = {}
    all_hypha_ids_seen = set()

    frame_numbers = [int(re.search(r"\d+", s).group()) for s in frame_folders]

    for frame_idx, frame_name in enumerate(frame_folders):
        frame_num = frame_numbers[frame_idx]
        frame_dir = os.path.join(DATASET_DIR, frame_name)
        print(f"\n==============================")
        print(f"Processing {frame_name}")
        print(f"Directory: {frame_dir}")
        print(f"==============================")

        pts_all = load_frame_points(frame_dir)
        if len(pts_all) == 0:
            print("No non-black pixels detected, skipping.")
            continue
        print(f"Total pixels detected: {len(pts_all)}")

        pts_ds = downsample_points(pts_all, DISPLAY_DOWNSAMPLE_N, seed=DISPLAY_DOWNSAMPLE_SEED + frame_idx)
        print(f"Downsampled for manual picking: {len(pts_ds)} points (target={DISPLAY_DOWNSAMPLE_N})")

        if FILTER_ENABLE:
            before = len(pts_ds)
            pts_ds = filter_points_knn_radius(
                pts_ds,
                radius=FILTER_NEIGHBOR_RADIUS,
                min_neighbors=FILTER_MIN_NEIGHBORS,
                k=FILTER_KNN
            )
            print(f"Filtered downsample: {before} -> {len(pts_ds)} points")

        if len(pts_ds) < 50:
            print("Too few points after downsample/filter, skipping.")
            continue

        picker = ManualPointPicker3D(pts_ds, title=f"{frame_name}: pick hyphae on global cloud", points_s=4, bg_alpha=0.05)
        assign, accepted, quit_all = picker.run()

        if quit_all:
            print("User quit.")
            break
        if not accepted:
            print("Selection not accepted; skipping this frame.")
            continue

        hypha_ids = sorted(set(assign) - {-1})
        if len(hypha_ids) == 0:
            print("No hypha points selected; skipping skeletonisation.")
            continue

        print(f"Selected hypha IDs in this frame: {hypha_ids}")
        frames_numeric.append(frame_num)

        skels_by_hypha = {}
        per_hypha_metrics_this_frame = {}

        for hid in hypha_ids:
            pts_h = pts_ds[assign == hid]
            print(f"  Hypha {hid}: {len(pts_h)} points -> skeletonising...")
            sk = skeletonise_cluster(pts_h)
            skels_by_hypha[hid] = sk

            # export assigned points
            pts_path = os.path.join(points_dir, f"frame_{frame_num:04d}_hypha_{hid}_points.csv")
            write_csv(pts_path, header=["x_pix", "y_pix", "z_pix"], rows=pts_h.tolist())

            nodes = sk.get("nodes", np.empty((0, 3)))
            edges = sk.get("edges", [])

            spur_nodes, spur_count = classify_spurs(nodes, edges, spur_max_nodes=SPUR_MAX_NODES)
            spur_nodes_count = int(len(spur_nodes))

            tips_no_spur = count_tips_excluding_spurs(nodes, edges, spur_nodes)
            branch_nodes_proper = count_branch_nodes_with_proper_branches(nodes, edges, spur_max_nodes=SPUR_MAX_NODES)
            total_nodes = int(len(nodes))
            quality = float((total_nodes - spur_nodes_count) / total_nodes) if total_nodes > 0 else 0.0

            length_um = skeleton_length_um(nodes, edges, um_per_pix, exclude_nodes=spur_nodes)

            # ---- UPDATED: volume with varying radius along skeleton (still excluding spur nodes)
            vol_um3 = hypha_volume_um3_variable_radius(
                pts_h, nodes, edges, um_per_pix,
                exclude_nodes=spur_nodes,
                smooth_iters=RADIUS_SMOOTH_ITERS
            )

            # export skeleton nodes + edges + spur flags
            nodes_path = os.path.join(skel_dir, f"frame_{frame_num:04d}_hypha_{hid}_nodes.csv")
            node_rows = []
            for i in range(len(nodes)):
                is_spur = 1 if i in spur_nodes else 0
                node_rows.append([i, nodes[i, 0], nodes[i, 1], nodes[i, 2], is_spur])
            write_csv(nodes_path, header=["node_id", "x_pix", "y_pix", "z_pix", "is_spur"], rows=node_rows)

            edges_path = os.path.join(skel_dir, f"frame_{frame_num:04d}_hypha_{hid}_edges.csv")
            write_csv(edges_path, header=["i", "j"], rows=[list(e) for e in edges])

            per_hypha_metrics_this_frame[hid] = {
                "total_nodes": total_nodes,
                "spur_nodes": spur_nodes_count,
                "spur_connections": int(spur_count),
                "quality": quality,
                "tips_no_spur": int(tips_no_spur),
                "branch_nodes_proper": int(branch_nodes_proper),
                "length_um_no_spur": float(length_um),
                "volume_um3_no_spur": float(vol_um3),
            }

            all_hypha_ids_seen.add(hid)

        # show overlay (unchanged)
        plot_skeletonised_overlay(
            frame_name=frame_name,
            global_points_xyz=pts_ds,
            assign=assign,
            skels_by_hypha=skels_by_hypha,
            title_suffix="(manual selection -> skeleton)",
            points_alpha=0.04,
            points_s=2,
            node_s=26
        )

        # global metrics
        global_total_nodes = 0
        global_spur_nodes = 0
        global_tips_no_spur = 0
        global_branch_nodes_proper = 0
        global_length_um_no_spur_sum = 0.0
        global_volume_um3_sumhypha = 0.0

        all_nonspur_skel_nodes = []

        for hid, sk in skels_by_hypha.items():
            nodes = sk.get("nodes", np.empty((0, 3)))
            edges = sk.get("edges", [])
            spur_nodes, _ = classify_spurs(nodes, edges, spur_max_nodes=SPUR_MAX_NODES)

            global_total_nodes += int(len(nodes))
            global_spur_nodes += int(len(spur_nodes))
            global_tips_no_spur += count_tips_excluding_spurs(nodes, edges, spur_nodes)
            global_branch_nodes_proper += count_branch_nodes_with_proper_branches(nodes, edges, spur_max_nodes=SPUR_MAX_NODES)
            global_length_um_no_spur_sum += skeleton_length_um(nodes, edges, um_per_pix, exclude_nodes=spur_nodes)
            global_volume_um3_sumhypha += per_hypha_metrics_this_frame[hid]["volume_um3_no_spur"]

            if len(nodes) > 0:
                keep_idx = [i for i in range(len(nodes)) if i not in spur_nodes]
                if len(keep_idx) > 0:
                    all_nonspur_skel_nodes.append(nodes[keep_idx])

        if len(all_nonspur_skel_nodes) > 0:
            all_nonspur_skel_nodes = np.vstack(all_nonspur_skel_nodes)
        else:
            all_nonspur_skel_nodes = np.empty((0, 3))

        global_quality = float((global_total_nodes - global_spur_nodes) / global_total_nodes) if global_total_nodes > 0 else 0.0

        # for global prism inflate radius: use median of per-hypha *median* radii (quick)
        hypha_radii_um = []
        for hid, sk in skels_by_hypha.items():
            nodes = sk.get("nodes", np.empty((0, 3)))
            pts_h = pts_ds[assign == hid]
            # reuse node radii estimation and take its median as hypha radius summary
            if len(nodes) >= 2 and len(pts_h) >= 5:
                r_nodes = estimate_node_radii_um(pts_h, nodes, um_per_pix, smooth_iters=0)
                if np.any(np.isfinite(r_nodes)) and float(np.nanmedian(r_nodes)) > 0:
                    hypha_radii_um.append(float(np.nanmedian(r_nodes)))

        inflate_r_um = float(np.median(hypha_radii_um)) if len(hypha_radii_um) else 0.0
        global_prism_vol = global_prism_volume_um3(all_nonspur_skel_nodes, um_per_pix, inflate_radius_um=inflate_r_um) if inflate_r_um > 0 else 0.0

        global_series["global_nodes"].append(int(global_total_nodes))
        global_series["global_branch_nodes_proper"].append(int(global_branch_nodes_proper))
        global_series["global_spur_nodes"].append(int(global_spur_nodes))
        global_series["global_quality"].append(float(global_quality))
        global_series["global_tips"].append(int(global_tips_no_spur))
        global_series["global_volume_prism"].append(float(global_prism_vol))
        global_series["global_volume_sumhypha"].append(float(global_volume_um3_sumhypha))

        global_rows.append([
            frame_num,
            global_total_nodes,
            global_spur_nodes,
            global_quality,
            global_branch_nodes_proper,
            global_tips_no_spur,
            global_length_um_no_spur_sum,
            global_prism_vol,
            global_volume_um3_sumhypha
        ])

        for hid in hypha_ids:
            m = per_hypha_metrics_this_frame[hid]
            hypha_rows.append([
                frame_num, hid,
                m["total_nodes"], m["spur_nodes"], m["spur_connections"],
                m["quality"], m["branch_nodes_proper"], m["tips_no_spur"],
                m["length_um_no_spur"], m["volume_um3_no_spur"]
            ])

        for hid in all_hypha_ids_seen:
            if hid not in hypha_series:
                hypha_series[hid] = {
                    "quality": [],
                    "branch_nodes_proper": [],
                    "tips_no_spur": [],
                    "length_um_no_spur": [],
                    "volume_um3_no_spur": [],
                }

        for hid in all_hypha_ids_seen:
            if hid in per_hypha_metrics_this_frame:
                m = per_hypha_metrics_this_frame[hid]
                hypha_series[hid]["quality"].append(m["quality"])
                hypha_series[hid]["branch_nodes_proper"].append(m["branch_nodes_proper"])
                hypha_series[hid]["tips_no_spur"].append(m["tips_no_spur"])
                hypha_series[hid]["length_um_no_spur"].append(m["length_um_no_spur"])
                hypha_series[hid]["volume_um3_no_spur"].append(m["volume_um3_no_spur"])
            else:
                hypha_series[hid]["quality"].append(float("nan"))
                hypha_series[hid]["branch_nodes_proper"].append(float("nan"))
                hypha_series[hid]["tips_no_spur"].append(float("nan"))
                hypha_series[hid]["length_um_no_spur"].append(float("nan"))
                hypha_series[hid]["volume_um3_no_spur"].append(float("nan"))

    print("\nAll done processing frames (or user quit).")

    global_csv = os.path.join(export_root, "global_metrics.csv")
    write_csv(
        global_csv,
        header=[
            "frame",
            "global_total_nodes",
            "global_spur_nodes",
            "global_quality",
            "global_branch_nodes_proper",
            "global_tips_no_spur",
            "global_total_length_um_no_spur_sum",
            "global_volume_prism_um3_no_spur",
            "global_volume_sumhypha_um3_no_spur"
        ],
        rows=global_rows
    )

    hypha_csv = os.path.join(export_root, "hypha_metrics.csv")
    write_csv(
        hypha_csv,
        header=[
            "frame", "hypha_id",
            "total_nodes",
            "spur_nodes",
            "spur_connections",
            "quality",
            "branch_nodes_proper",
            "tips_no_spur",
            "length_um_no_spur",
            "volume_um3_no_spur"
        ],
        rows=hypha_rows
    )

    print(f"\nWrote CSVs:\n  {global_csv}\n  {hypha_csv}")
    print(f"Wrote per-frame point clouds to: {points_dir}")
    print(f"Wrote per-frame skeleton nodes/edges to: {skel_dir}")

    if len(frames_numeric) == 0:
        print("No frames were accepted; no plots to show.")
        return

    # 1) length per hypha (excluding spurs)
    plot_subplots_per_hypha(
        frames_numeric,
        sorted(all_hypha_ids_seen),
        {hid: {"length_um_no_spur": hypha_series[hid]["length_um_no_spur"]} for hid in all_hypha_ids_seen},
        title="(1) Hypha length over frames (µm), excluding spurs",
        ylabels=["length_um_no_spur"],
        ncols=3
    )

    # 2) global nodes
    plot_time_series(frames_numeric, global_series["global_nodes"],
                     title="(2) Global total nodes over frames",
                     ylabel="Nodes")

    # 3) global proper branching nodes
    plot_time_series(frames_numeric, global_series["global_branch_nodes_proper"],
                     title="(3) Global branching nodes with proper branches over frames",
                     ylabel="Count")

    # 4) global quality
    plot_time_series(frames_numeric, global_series["global_quality"],
                     title="(4) Global network quality over frames (1 - spur_nodes/total_nodes)",
                     ylabel="Quality")

    # 5) per-hypha quality + branches
    plot_subplots_per_hypha(
        frames_numeric,
        sorted(all_hypha_ids_seen),
        {hid: {
            "quality": hypha_series[hid]["quality"],
            "branch_nodes_proper": hypha_series[hid]["branch_nodes_proper"]
        } for hid in all_hypha_ids_seen},
        title="(5) Per-hypha quality + branch nodes (proper) over frames",
        ylabels=["quality", "branch_nodes_proper"],
        ncols=3
    )

    # 6) global tips (excluding spurs)
    plot_time_series(frames_numeric, global_series["global_tips"],
                     title="(6) Global tips over frames (excluding spurs)",
                     ylabel="Tips")

    # 7) per-hypha tips (excluding spurs)
    plot_subplots_per_hypha(
        frames_numeric,
        sorted(all_hypha_ids_seen),
        {hid: {"tips_no_spur": hypha_series[hid]["tips_no_spur"]} for hid in all_hypha_ids_seen},
        title="(7) Per-hypha tips over frames (excluding spurs)",
        ylabels=["tips_no_spur"],
        ncols=3
    )

    # 8) global prism volume
    if _HAS_HULL:
        plot_time_series(frames_numeric, global_series["global_volume_prism"],
                         title="(8) Global prism volume over frames (µm³), excluding spurs",
                         ylabel="Volume (µm³)")
    else:
        print("NOTE: scipy.spatial.ConvexHull not available; skipping plot (8).")

    # 9) total hypha volume and per-hypha volume (varying radius)
    plot_time_series(frames_numeric, global_series["global_volume_sumhypha"],
                     title="(9a) Total hypha volume (sum of per-hypha varying-radius volumes) over frames (µm³), excluding spurs",
                     ylabel="Volume (µm³)")

    plot_subplots_per_hypha(
        frames_numeric,
        sorted(all_hypha_ids_seen),
        {hid: {"volume_um3_no_spur": hypha_series[hid]["volume_um3_no_spur"]} for hid in all_hypha_ids_seen},
        title="(9b) Per-hypha volume over frames (µm³), excluding spurs (varying radius along skeleton)",
        ylabels=["volume_um3_no_spur"],
        ncols=3
    )

    print("\nFinished plots + exports.")

if __name__ == "__main__":
    main()
