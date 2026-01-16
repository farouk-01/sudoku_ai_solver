import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
from collections import defaultdict
import random

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

def show_all_container_assignments(assign, reps, hidden_groups, max_show=30, per_row=10):
    for cid in sorted(hidden_groups.keys()):
        members = assign.get(cid, [])
        if not members:
            continue

        container = reps[cid]

        members = [g for g in members if not np.array_equal(g, container)]

        grids = [container] + members
        if max_show is not None:
            grids = grids[:max_show + 1] 

        print(f"\n---- CONTAINER {cid} | total members: {len(members)} ----")
        show_patterns(grids, per_row=per_row)

def draw_hasse(covers_out, ranks, title="Hasse diagram (hide relation)", seed=0):
    """
    covers_out: dict[i] -> list[j] edges i -> j (i covers j)
    ranks: list[int] (node layers)
    """
    random.seed(seed)
    n = len(ranks)

    # Layer nodes by rank
    layers = defaultdict(list)
    for i, r in enumerate(ranks):
        layers[r].append(i)

    unique_ranks = sorted(layers.keys())

    # Assign x positions within each rank
    pos = {}
    for r in unique_ranks:
        nodes = layers[r]
        # stable random-ish spread
        xs = list(range(len(nodes)))
        random.shuffle(xs)
        for x, i in zip(xs, nodes):
            pos[i] = (x, r)

    # Normalize x within each layer for nicer spacing
    for r in unique_ranks:
        nodes = layers[r]
        if len(nodes) <= 1:
            for i in nodes:
                x, y = pos[i]
                pos[i] = (0.0, y)
        else:
            for k, i in enumerate(sorted(nodes)):
                pos[i] = (k / (len(nodes) - 1), r)

    plt.figure(figsize=(12, 6))

    # Edges
    for i, js in covers_out.items():
        xi, yi = pos[i]
        for j in js:
            xj, yj = pos[j]
            plt.plot([xi, xj], [yi, yj], lw=0.6)

    # Nodes
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    plt.scatter(xs, ys, s=18)

    plt.title(title)
    plt.xlabel("within-layer position")
    plt.ylabel("|motif_set|  (rank)")
    plt.tight_layout()
    plt.show()