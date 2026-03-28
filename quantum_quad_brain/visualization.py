"""
visualization.py
----------------
Visualization of 4D vertex mechanics and triple-folding operations.

Provides:
- plot_topology_2d      : 2D projection of 4D vertex graph
- plot_mesh_layers      : heatmap of triple-fold mesh layers
- plot_inference_history: probability / coherence timeline
- plot_fabric_entropy   : fabric-layer entropy chart
- save_all              : save all figures to disk
"""

from __future__ import annotations

import os
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if TYPE_CHECKING:
    from .quantum_quad_brain import QuantumQuadBrain
    from .inference import InferenceResult


def plot_topology_2d(
    brain: "QuantumQuadBrain",
    projection: tuple = (0, 1),
    ax: Optional[plt.Axes] = None,
    title: str = "4D Topology (2D Projection)",
) -> plt.Figure:
    """
    Plot a 2D projection of the 4D topological vertex graph.

    Parameters
    ----------
    projection : tuple of two ints
        Which two of the four dimensions to project onto (e.g. (0,1) → w,x axes).
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    topo = brain.topology
    coords = topo.coordinate_matrix()  # (N, 4)
    amps = topo.amplitude_vector()     # (N,) magnitudes

    xi, yi = projection
    xs = coords[:, xi]
    ys = coords[:, yi]

    # Draw edges
    for (a, b), weight in topo.edges.items():
        ax.plot(
            [topo.vertices[a].coords[xi], topo.vertices[b].coords[xi]],
            [topo.vertices[a].coords[yi], topo.vertices[b].coords[yi]],
            color="steelblue",
            alpha=min(weight / (max(topo.edges.values()) + 1e-9), 0.6),
            linewidth=0.8,
        )

    # Draw vertices, size / colour by amplitude
    sc = ax.scatter(
        xs, ys,
        c=amps,
        s=80 + 200 * amps / (amps.max() + 1e-9),
        cmap="plasma",
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
    )
    if fig is not None:
        plt.colorbar(sc, ax=ax, label="Amplitude magnitude")

    dim_labels = ["w", "x", "y", "z"]
    ax.set_xlabel(f"{dim_labels[xi]}-axis (dim {xi})")
    ax.set_ylabel(f"{dim_labels[yi]}-axis (dim {yi})")
    ax.set_title(title)
    ax.set_facecolor("#0d0d1a")

    return fig or ax.get_figure()


def plot_mesh_layers(
    brain: "QuantumQuadBrain",
    ax: Optional[plt.Axes] = None,
    title: str = "Triple-Fold Mesh Layers (Amplitude)",
) -> plt.Figure:
    """Heatmap grid of the three mesh layer amplitude distributions."""
    fig = None
    if ax is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        axes = [ax]

    mesh = brain.mesh
    for idx, (layer, subplot_ax) in enumerate(zip(mesh.layers, axes)):
        amp = np.abs(layer.cells)
        im = subplot_ax.imshow(amp, cmap="inferno", aspect="auto")
        subplot_ax.set_title(f"Layer {idx} (fold #{mesh.fold_count})")
        subplot_ax.set_xlabel("cols")
        subplot_ax.set_ylabel("rows")
        if fig is not None:
            plt.colorbar(im, ax=subplot_ax, label="|ψ|")

    if fig is not None:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout()
    return fig or axes[0].get_figure()


def plot_inference_history(
    history: "List[InferenceResult]",
    ax: Optional[plt.Axes] = None,
    title: str = "Inference History",
) -> plt.Figure:
    """Plot decision index and coherence over inference cycles."""
    if not history:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No inference data yet", ha="center", va="center")
        return fig

    cycles = [r.cycle for r in history]
    decisions = [r.decision for r in history]
    coherences = [r.coherence for r in history]
    probs = [r.probability for r in history]

    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    else:
        axes = [ax]
        fig = None

    ax0 = axes[0]
    ax0.step(cycles, decisions, where="mid", color="cyan", label="Decision index")
    ax0.set_ylabel("Decision (state index)")
    ax0.legend(loc="upper right")
    ax0.grid(alpha=0.3)
    ax0.set_facecolor("#0d0d1a")

    ax1 = axes[1]
    ax1.plot(cycles, coherences, color="lime", label="SUSY coherence")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Coherence")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_facecolor("#0d0d1a")

    ax2 = axes[2]
    ax2.plot(cycles, probs, color="orange", label="Decision probability")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Inference cycle")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#0d0d1a")

    if fig is not None:
        fig.suptitle(title, fontsize=14)
        fig.patch.set_facecolor("#0d0d1a")
        fig.tight_layout()
    return fig or ax.get_figure()


def plot_fabric_entropy(
    brain: "QuantumQuadBrain",
    ax: Optional[plt.Axes] = None,
    title: str = "Quantum Fabric Layer Entropy",
) -> plt.Figure:
    """Bar chart of total entropy for each quantum fabric layer."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    layers = brain.fabric_layers
    ids = [fl.layer_id for fl in layers]
    entropies = [fl.total_entropy() for fl in layers]
    ax.bar(ids, entropies, color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Fabric layer index")
    ax.set_ylabel("Total entropy (bits)")
    ax.set_title(title)
    ax.set_facecolor("#0d0d1a")
    if fig is not None:
        fig.patch.set_facecolor("#0d0d1a")
        fig.tight_layout()
    return fig or ax.get_figure()


def plot_dashboard(
    brain: "QuantumQuadBrain",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a comprehensive 2×3 dashboard combining all visualizations.
    """
    fig = plt.figure(figsize=(20, 14), facecolor="#0d0d1a")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0: topology projections (two 4D planes)
    ax_topo_wx = fig.add_subplot(gs[0, :2])
    plot_topology_2d(brain, projection=(0, 1), ax=ax_topo_wx, title="4D Topology: w-x plane")

    ax_topo_yz = fig.add_subplot(gs[0, 2:])
    plot_topology_2d(brain, projection=(2, 3), ax=ax_topo_yz, title="4D Topology: y-z plane")

    # Row 1: mesh layers
    for i, layer in enumerate(brain.mesh.layers):
        ax = fig.add_subplot(gs[1, i])
        amp = np.abs(layer.cells)
        im = ax.imshow(amp, cmap="inferno", aspect="auto")
        ax.set_title(f"Mesh Layer {i}", color="white")
        ax.tick_params(colors="white")
        plt.colorbar(im, ax=ax, label="|ψ|")

    # Row 1 col 3: fabric entropy
    ax_ent = fig.add_subplot(gs[1, 3])
    plot_fabric_entropy(brain, ax=ax_ent)
    ax_ent.tick_params(colors="white")
    ax_ent.title.set_color("white")

    # Row 2: inference history
    if brain.pipeline.history:
        ax_inf = fig.add_subplot(gs[2, :])
        h = brain.pipeline.history
        cycles = [r.cycle for r in h]
        decisions = [r.decision for r in h]
        coherences = [r.coherence for r in h]

        ax_inf2 = ax_inf.twinx()
        ax_inf.step(cycles, decisions, where="mid", color="cyan", label="Decision", alpha=0.8)
        ax_inf2.plot(cycles, coherences, color="lime", label="Coherence", alpha=0.8)
        ax_inf.set_xlabel("Cycle", color="white")
        ax_inf.set_ylabel("Decision state", color="cyan")
        ax_inf2.set_ylabel("Coherence", color="lime")
        ax_inf.set_facecolor("#0d0d1a")
        ax_inf.tick_params(colors="white")
        ax_inf2.tick_params(colors="white")
        ax_inf.set_title("Inference History: Decision & Coherence", color="white")

    # Style all axes
    for ax in fig.get_axes():
        ax.set_facecolor("#0d0d1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    fig.suptitle(
        "Quantum Quad-Brain System — Live Dashboard",
        fontsize=18,
        color="white",
        y=1.01,
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def save_all(brain: "QuantumQuadBrain", output_dir: str = "output") -> List[str]:
    """Save all individual visualizations plus the dashboard to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    def _save(fig: plt.Figure, name: str) -> str:
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return path

    saved.append(_save(plot_topology_2d(brain, projection=(0, 1)), "topology_wx.png"))
    saved.append(_save(plot_topology_2d(brain, projection=(2, 3)), "topology_yz.png"))
    saved.append(_save(plot_mesh_layers(brain), "mesh_layers.png"))
    saved.append(_save(plot_inference_history(brain.pipeline.history), "inference_history.png"))
    saved.append(_save(plot_fabric_entropy(brain), "fabric_entropy.png"))
    saved.append(_save(plot_dashboard(brain), "dashboard.png"))

    return saved
