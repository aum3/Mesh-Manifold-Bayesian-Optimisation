"""
mesh_bo_utils.py
================
Support code for the Mesh Bayesian Optimisation notebook.

Everything in here is plumbing: 3D mesh plotting, a small PLY reader, and a
couple of geometry helpers. The actual method (kernel setup, objective, expected
improvement, GP posterior, BO loop) lives in the notebook on purpose.

Conventions
-----------
* Vertices are indexed from 0 everywhere (numpy style).
* ``mesh`` is a ``geometric_kernels.spaces.Mesh``; only ``.vertices``,
  ``.faces`` and ``.num_vertices`` are used here.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SAMPLED_COLOUR = "#008CFF"   # points BO has already evaluated
NEXT_COLOUR = "red"          # point the acquisition function wants next
VIEWS = {"front": (20, 35), "back": (-20, 215)}   # (elev, azim)


# ---------------------------------------------------------------------------
# housekeeping
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 0) -> np.random.Generator:
    """Seed the global RNGs and hand back a numpy Generator to use directly."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def apply_plot_style() -> None:
    """Global matplotlib defaults so every figure in the notebook matches."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _centre(mesh) -> np.ndarray:
    return np.asarray(mesh.vertices)[:, :3].mean(axis=0)


def _camera_direction(ax) -> np.ndarray:
    """Unit vector from the origin towards the camera for a 3D axis."""
    el, az = np.radians(ax.elev), np.radians(ax.azim)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def facing_camera(ax, mesh, points) -> np.ndarray:
    """Boolean mask: which points are on the near side of the mesh right now."""
    return (np.atleast_2d(points) - _centre(mesh)) @ _camera_direction(ax) > 0


def view_towards(ax, mesh, point) -> None:
    """Rotate the camera so `point` is roughly front and centre."""
    d = np.asarray(point, dtype=float).reshape(-1)[:3] - _centre(mesh)
    d /= np.linalg.norm(d) + 1e-12
    ax.view_init(elev=np.degrees(np.arcsin(d[2])), azim=np.degrees(np.arctan2(d[1], d[0])))


def surface_path(mesh, points, n_per_segment: int = 40, lift: float = 1.01) -> np.ndarray:
    """
    Join consecutive points with arcs that hug the surface.

    Straight lines would cut through the mesh, so each segment is interpolated in
    (direction, radius) space around the mesh centre, then lifted by ~1% so the
    curve isn't z-fighting with the faces.
    """
    centre = _centre(mesh)
    rel = np.asarray(points, dtype=float) - centre
    radii = np.linalg.norm(rel, axis=1)
    t = np.linspace(0, 1, n_per_segment)[:, None]
    segments = []
    for a in range(len(rel) - 1):
        seg = (1 - t) * rel[a] + t * rel[a + 1]
        direction = seg / (np.linalg.norm(seg, axis=1, keepdims=True) + 1e-12)
        radius = (1 - t) * radii[a] + t * radii[a + 1]
        segments.append(centre + lift * direction * radius)
    return np.vstack(segments)


def nearest_vertex_indices(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    For each row of `points` (assumed to live on the unit sphere) return the index
    of the closest mesh vertex after projecting the vertices to the unit sphere.
    Output has shape (n, 1) and dtype int64, which is what the mesh kernel wants.
    """
    verts = np.asarray(vertices)[:, :3]
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    _, idx = cKDTree(verts).query(points)
    return idx.astype(np.int64).reshape(-1, 1)


def geodesic_distances(unit_points: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances between points on the unit sphere."""
    gram = np.clip(unit_points @ unit_points.T, -1.0, 1.0)   # clip: rounding can push |dot| past 1
    return np.arccos(gram)


# ---------------------------------------------------------------------------
# PLY reader
# ---------------------------------------------------------------------------
_PLY_DTYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}


def read_ply_vertices(path) -> np.ndarray:
    """
    Pull the raw (x, y, z) vertex positions out of a PLY file (ascii or binary).

    geometric_kernels already parses the mesh for us, but I wanted the untouched
    point cloud to compare against, and this avoids pulling in another dependency.
    Faces and any other elements are ignored.
    """
    with open(Path(path), "rb") as fh:
        fmt, n_verts, props, in_vertex = "ascii", 0, [], False

        # header
        while True:
            raw = fh.readline()
            if not raw:
                raise ValueError(f"{path}: reached EOF before 'end_header'")
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_verts, in_vertex = int(line.split()[2]), True
            elif line.startswith("element "):
                in_vertex = False
            elif line.startswith("property ") and in_vertex:
                _, dtype, name = line.split()[:3]
                props.append((name, dtype))
            elif line == "end_header":
                break

        # body
        if fmt == "ascii":
            return np.loadtxt(fh, max_rows=n_verts)[:, :3]

        endian = "<" if "little" in fmt else ">"
        dt = np.dtype([(name, endian + _PLY_DTYPES[t]) for name, t in props])
        raw = np.frombuffer(fh.read(n_verts * dt.itemsize), dtype=dt, count=n_verts)
        return np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)


# ---------------------------------------------------------------------------
# low-level mesh plotting
# ---------------------------------------------------------------------------
def _face_shading(verts, faces, light=(0.4, -0.3, 0.85)) -> np.ndarray:
    """Fake diffuse lighting so the surface reads as 3D instead of a flat blob."""
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.asarray(light, dtype=float)
    light /= np.linalg.norm(light)
    return 0.55 + 0.45 * np.abs(normals @ light)


def plot_mesh_values(ax, mesh, values, cmap="viridis", vmin=None, vmax=None,
                     title=None, elev=20, azim=35):
    """
    Paint a scalar field (one value per vertex) onto the mesh.

    Returns a ScalarMappable so the caller can attach a colourbar.
    """
    verts = np.asarray(mesh.vertices)[:, :3]
    faces = np.asarray(mesh.faces)
    values = np.asarray(values, dtype=float).reshape(-1)

    face_vals = values[faces].mean(axis=1)            # one colour per triangle
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    if vmax <= vmin:                                   # constant field -> avoid zero-width range
        vmax = vmin + 1e-12
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl.colormaps[cmap]

    colours = cmap_obj(norm(face_vals))
    colours[:, :3] *= _face_shading(verts, faces)[:, None]

    ax.computed_zorder = False                         # respect zorder so markers stay on top
    surface = Poly3DCollection(verts[faces], facecolors=colours, linewidths=0, edgecolors="none")
    surface.set_zorder(1)
    ax.add_collection3d(surface)

    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.35)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)


def add_markers(ax, mesh, points, hide_back=True, **scatter_kwargs):
    """Scatter 3D points on top of the mesh, optionally dropping the ones on the far side."""
    points = np.atleast_2d(points)
    if hide_back:
        points = points[facing_camera(ax, mesh, points)]
    if len(points) == 0:
        return None
    scatter_kwargs.setdefault("depthshade", False)
    scatter_kwargs.setdefault("zorder", 10)
    return ax.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_kwargs)


# ---------------------------------------------------------------------------
# figures used in the notebook (one per section)
# ---------------------------------------------------------------------------
def plot_objective(mesh, values, cmap="hot"):
    """Objective on the mesh, front and back."""
    fig = plt.figure(figsize=(11, 5))
    for n, (side, (elev, azim)) in enumerate(VIEWS.items(), start=1):
        ax = fig.add_subplot(1, 2, n, projection="3d")
        sm = plot_mesh_values(ax, mesh, values, cmap=cmap, elev=elev, azim=azim,
                              title=f"Objective function ({side})")
    fig.colorbar(sm, ax=fig.axes, shrink=0.7, pad=0.02, label="objective value")
    return fig


def plot_posterior_evolution(mesh, history, cmap="viridis"):
    """
    GP posterior mean after 1, 2, ... k sample points. Top row front, bottom row back.

    `history[i]` needs `.mean` (posterior mean per vertex) and `.x_obs` (vertex indices,
    shape (i + 2, 1): the i + 1 conditioned-on points plus the one EI proposes next).
    All panels share one colour scale so they're directly comparable.
    """
    k_max = len(history)
    all_means = np.hstack([step.mean for step in history])
    vmin, vmax = all_means.min(), all_means.max()

    fig = plt.figure(figsize=(3.3 * k_max, 7.5))
    for row, (side, (elev, azim)) in enumerate(VIEWS.items()):
        for k, step in enumerate(history, start=1):
            ax = fig.add_subplot(2, k_max, row * k_max + k, projection="3d")
            title = f"After {k} sample point{'s' if k > 1 else ''}" if row == 0 else None
            sm = plot_mesh_values(ax, mesh, step.mean, cmap=cmap, vmin=vmin, vmax=vmax,
                                  title=title, elev=elev, azim=azim)
            xs = step.x_obs.ravel()
            add_markers(ax, mesh, mesh.vertices[xs[:-1]][:, :3],
                        s=45, c=SAMPLED_COLOUR, edgecolors="white")
            add_markers(ax, mesh, mesh.vertices[xs[-1:]][:, :3],
                        s=70, c=NEXT_COLOUR, marker="D", edgecolors="white")
            if k == 1:
                ax.text2D(-0.05, 0.5, side, transform=ax.transAxes,
                          rotation=90, va="center", fontsize=11)

    fig.subplots_adjust(left=0.02, right=0.92, top=0.9, bottom=0.08, wspace=0.0, hspace=-0.05)
    fig.suptitle("GP posterior mean over the mesh as BO proceeds", fontsize=15)
    fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.01, label="posterior mean")
    fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color=SAMPLED_COLOUR, markeredgecolor="white",
                       markersize=9, label="sampled point"),
            plt.Line2D([], [], marker="D", ls="", color=NEXT_COLOUR, markeredgecolor="white",
                       markersize=9, label="next point (max EI)"),
        ],
        loc="lower center", ncol=2, fontsize=11,
    )
    return fig


def plot_sample_path(mesh, objective_vals, posterior_mean, x_obs, posterior_label="Posterior mean"):
    """Objective next to a posterior mean, with the sampling order drawn as a path on the surface."""
    path_points = mesh.vertices[np.asarray(x_obs).ravel()][:, :3]
    path_curve = surface_path(mesh, path_points)
    panels = [("Objective", objective_vals, "hot"), (posterior_label, posterior_mean, "viridis")]

    fig = plt.figure(figsize=(12, 10))
    for r, (side, (elev, azim)) in enumerate(VIEWS.items()):
        for c, (title, vals, cmap) in enumerate(panels):
            ax = fig.add_subplot(2, 2, r * 2 + c + 1, projection="3d")
            sm = plot_mesh_values(ax, mesh, vals, cmap=cmap, elev=elev, azim=azim,
                                  title=f"{title} ({side})")
            fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)

            # break the line wherever it wraps round to the far side
            curve = path_curve.copy()
            curve[~facing_camera(ax, mesh, curve)] = np.nan
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=SAMPLED_COLOUR, linewidth=2.5, zorder=9)

            add_markers(ax, mesh, path_points, s=80, c=SAMPLED_COLOUR, edgecolors="white")
            visible = facing_camera(ax, mesh, path_points)
            for i, p in enumerate(path_points):
                if visible[i]:
                    ax.text(p[0], p[1], p[2], f" {i + 1} ", fontsize=11, fontweight="bold", zorder=11,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    fig.subplots_adjust(wspace=0.05, hspace=0.0)
    fig.suptitle("Sample path (numbers show sampling order)", fontsize=14)
    return fig


def plot_kernel_influence(mesh, kernel, params, source_vertex: int):
    """Heat-map of k(source, .) over the whole mesh."""
    source = np.array([[source_vertex]])
    everything = np.arange(mesh.num_vertices).reshape(-1, 1)
    k_source = kernel.K(params, source, everything).ravel()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    sm = plot_mesh_values(ax, mesh, k_source, cmap="viridis",
                          title=f"Kernel influence from vertex {source_vertex}")
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, label="kernel value")

    source_xyz = mesh.vertices[source_vertex][:3]
    view_towards(ax, mesh, source_xyz)                 # turn to face the source so it's visible
    add_markers(ax, mesh, source_xyz, s=150, c=NEXT_COLOUR, marker="D",
                edgecolors="white", label="source vertex")
    ax.legend(loc="lower left")
    return fig


def plot_kernel_comparison(geo_dist, mesh_K, sphere_K, n_pairs=50_000, seed=0):
    """
    Kernel value vs geodesic distance for the mesh kernel and the continuous sphere kernel.

    All n^2 pairs is ~16M points per series and just turns into a solid blob, so a
    random subset of pairs is plotted instead.
    """
    rng = np.random.default_rng(seed)
    n = geo_dist.shape[0]
    i, j = rng.integers(0, n, size=(2, n_pairs))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(geo_dist[i, j], mesh_K[i, j], c="tab:red", s=4, alpha=0.5,
               label="Mesh (discretised) kernel")
    ax.scatter(geo_dist[i, j], sphere_K[i, j], c="tab:blue", s=4, alpha=0.5,
               label="Continuous sphere kernel")
    ax.set_xlabel("Geodesic distance")
    ax.set_ylabel("Kernel value")
    ax.set_title("Mesh kernel vs continuous kernel")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", markerscale=3)
    fig.tight_layout()
    return fig
