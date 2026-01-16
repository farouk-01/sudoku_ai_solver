import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from collections import defaultdict

def show_patterns(grids, per_row=10):
    if isinstance(grids, np.ndarray) and grids.ndim == 2:
        grids = [grids]
    n = len(grids)
    rows = math.ceil(n / per_row)

    fig, axes = plt.subplots(rows, per_row, figsize=(per_row*2, rows*2))
    axes = np.atleast_2d(axes)

    num_colors = {
        1: "red",
        2: "#1591EA",
        3: "green",
        4: "purple",
    }

    for idx, grid in enumerate(grids):
        ax = axes[idx // per_row, idx % per_row]
        ax.set_aspect("equal")
        ax.set_xlim(0, 4); ax.set_ylim(0, 4)
        ax.invert_yaxis()

        for i in range(5):
            ax.plot([0, 4], [i, i], lw=1)
            ax.plot([i, i], [0, 4], lw=1)

        for r in range(4):
            for c in range(4):
                v = grid[r, c]
                if v:
                    ax.text(c+0.5, r+0.5, str(v),
                            ha="center", va="center", fontsize=14)

        pos = {}
        for r in range(4):
            for c in range(4):
                v = grid[r, c]
                if v:
                    pos.setdefault(v, []).append((r, c))

        for v, cells in pos.items():
            color = num_colors.get(v, "black")
            for (r1,c1),(r2,c2) in combinations(cells, 2):
                if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                    ax.plot([c1+0.5, c2+0.5],
                            [r1+0.5, r2+0.5],
                            color=color, lw=4)

        ax.axis("off")

    for i in range(n, rows*per_row):
        axes[i // per_row, i % per_row].axis("off")

    plt.tight_layout()
    plt.show()


