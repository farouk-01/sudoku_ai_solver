import numpy as np
from visual_tools import *
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
import math


def normalize_digits(grid):
    """
    ordre d'apparition:
    ex: [4,3,4,1] -> [1,2,1,3]
    """
    mapping = {}
    nxt = 1
    out = grid.copy()
    for r in range(4):
        for c in range(4):
            v = int(out[r, c])
            if v not in mapping:
                mapping[v] = nxt
                nxt += 1
            out[r, c] = mapping[v]
    return out

def symmetries(grid):
    g = grid
    for k in range(4):
        r = np.rot90(g, k)
        yield r
        yield np.fliplr(r)

def canonical_key(grid):
    keys = []
    for g in symmetries(grid):
        gn = normalize_digits(g)
        keys.append(tuple(gn.flatten().tolist()))
    return min(keys)

def classify(grids):
    groups = {}
    for i, g in enumerate(grids):
        key = canonical_key(g)
        groups.setdefault(key, []).append(g)
    return groups


def show_group(groups, key, max_show=None, per_row=10, shuffle=False, seed=0):
    grids = groups[key]
    if shuffle:
        rng = np.random.default_rng(seed)
        grids = [grids[i] for i in rng.permutation(len(grids))]
    if max_show is not None:
        grids = grids[:max_show]
    show_patterns(grids, per_row=per_row)

def show_largest_groups(groups, top=5):
    sizes = sorted([(len(v), k) for k, v in groups.items()], reverse=True)
    for i in range(min(top, len(sizes))):
        print(f"#{i+1}: taille={sizes[i][0]}")
    return sizes 

def find_solutions_with_two_cols(solutions, col1, col2, allow_reversed=True):
    c1 = tuple(int(x) for x in col1)
    c2 = tuple(int(x) for x in col2)
    c1r = c1[::-1]
    c2r = c2[::-1]

    matches = []
    for g in solutions:
        # Extract the 4 columns as tuples
        cols = [tuple(int(x) for x in g[:, j]) for j in range(4)]
        colset = set(cols)

        ok1 = (c1 in colset) or (allow_reversed and c1r in colset)
        ok2 = (c2 in colset) or (allow_reversed and c2r in colset)

        if ok1 and ok2:
            matches.append(g)

    return matches


def find_solutions_with_cols_at_positions(
    solutions,
    col1, idx1,
    col2, idx2,
    allow_reversed=True
):
    c1 = tuple(int(x) for x in col1)
    c2 = tuple(int(x) for x in col2)
    c1r = c1[::-1]
    c2r = c2[::-1]

    out = []
    for g in solutions:
        col_at_1 = tuple(int(x) for x in g[:, idx1])
        col_at_2 = tuple(int(x) for x in g[:, idx2])

        ok1 = (col_at_1 == c1) or (allow_reversed and col_at_1 == c1r)
        ok2 = (col_at_2 == c2) or (allow_reversed and col_at_2 == c2r)

        if ok1 and ok2:
            out.append(g)

    return out

def find_solutions_with_adjacent_cols(
    solutions,
    col1,
    col2,
    allow_reversed=True,
    order_matters=False,
    start_col=None
):
    c1 = tuple(int(x) for x in col1)
    c2 = tuple(int(x) for x in col2)
    c1r = c1[::-1]
    c2r = c2[::-1]

    def matches(a, b):
        ok1 = (a == c1) or (allow_reversed and a == c1r)
        ok2 = (b == c2) or (allow_reversed and b == c2r)
        return ok1 and ok2

    if start_col is not None and start_col not in (0, 1, 2):
        raise ValueError("start_col must be None or 0/1/2 (for pairs (0,1), (1,2), (2,3))")

    out = []
    for g in solutions:
        cols = [tuple(int(x) for x in g[:, j]) for j in range(4)]

        js = [start_col] if start_col is not None else [0, 1, 2]
        for j in js:
            left, right = cols[j], cols[j + 1]

            if matches(left, right) or (not order_matters and matches(right, left)):
                out.append(g)
                break

    return out

import numpy as np

def find_solutions_with_adjacent_rows(
    solutions,
    row1,
    row2,
    allow_reversed=True,
    order_matters=False,
    start_row=None 
):
    r1 = tuple(int(x) for x in row1)
    r2 = tuple(int(x) for x in row2)
    r1r = r1[::-1]
    r2r = r2[::-1]

    def matches(a, b):
        ok1 = (a == r1) or (allow_reversed and a == r1r)
        ok2 = (b == r2) or (allow_reversed and b == r2r)
        return ok1 and ok2

    if start_row is not None and start_row not in (0, 1, 2):
        raise ValueError("start_row must be None or 0/1/2 (pairs (0,1), (1,2), (2,3))")

    out = []
    for g in solutions:
        rows = [tuple(int(x) for x in g[i, :]) for i in range(4)]

        iseq = [start_row] if start_row is not None else [0, 1, 2]
        for i in iseq:
            top, bottom = rows[i], rows[i + 1]

            if matches(top, bottom) or (not order_matters and matches(bottom, top)):
                out.append(g)
                break

    return out

def motif_features(grid):
    diag_adj = 0
    x_blocks = 0
    for r in range(4):
        for c in range(4):
            v = grid[r, c]
            if v == 0:
                continue
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                r2, c2 = r+dr, c+dc
                if 0 <= r2 < 4 and 0 <= c2 < 4:
                    if grid[r2, c2] == v:
                        diag_adj += 1
    diag_adj //= 2 
    for br in (0, 2):
        for bc in (0, 2):
            a = grid[br, bc]
            b = grid[br+1, bc+1]
            c = grid[br, bc+1]
            d = grid[br+1, bc]
            if a == b and c == d:
                x_blocks += 1

    return np.array([diag_adj, x_blocks])

def group_by_cluster(sols, labels):
    clusters = defaultdict(list)
    for g, lab in zip(sols, labels):
        clusters[int(lab)].append(g)
    return dict(clusters)

def show_cluster_reps(sols, labels, per_row=10, rep="first"):
    clusters = group_by_cluster(sols, labels)

    reps = []
    cluster_ids = sorted(clusters.keys())
    for cid in cluster_ids:
        if rep == "first":
            reps.append(clusters[cid][0])
        elif rep == "canonical":
            reps.append(min(clusters[cid], key=lambda g:canonical_key(g)))
        else:
            raise ValueError("rep must be 'first' or 'canonical'")

    print(f"Clusters: {len(reps)} (showing 1 representative each)")
    show_patterns(reps, per_row=per_row)
    return clusters, cluster_ids

def show_cluster(clusters, cid, per_row=10, max_show=None, shuffle=False, seed=0):
    grids = clusters[int(cid)]
    if shuffle:
        import numpy as np
        rng = np.random.default_rng(seed)
        grids = [grids[i] for i in rng.permutation(len(grids))]
    if max_show is not None:
        grids = grids[:max_show]

    print(f"Cluster {cid}: {len(clusters[int(cid)])} grids (showing {len(grids)})")
    show_patterns(grids, per_row=per_row)

def show_clusters_by_size(sols, labels, per_row=10, max_clusters=None, rep="canonical"):
    clusters = group_by_cluster(sols, labels)
    order = sorted(clusters.keys(), key=lambda k: len(clusters[k]), reverse=True)
    if max_clusters is not None:
        order = order[:max_clusters]

    reps = []
    for cid in order:
        if rep == "first":
            reps.append(clusters[cid][0])
        else:
            reps.append(min(clusters[cid], key=lambda g: canonical_key(g)))

    print("Top clusters by size:", [(cid, len(clusters[cid])) for cid in order[:10]])
    show_patterns(reps, per_row=per_row)
    return clusters, order

def motif_diag_adj_pairs(grid):
    pairs = set()
    for r in range(4):
        for c in range(4):
            v = int(grid[r, c])
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 4 and 0 <= c2 < 4 and int(grid[r2, c2]) == v:
                    a, b = (r, c), (r2, c2)
                    pairs.add(tuple(sorted((a, b))))
    return pairs

def motif_xblock_positions(grid):
    out = set()
    for br in (0, 2):
        for bc in (0, 2):
            if int(grid[br, bc]) == int(grid[br+1, bc+1]) and int(grid[br, bc+1]) == int(grid[br+1, bc]):
                out.add((br, bc))
    return out

def motif_set(grid):
    s = set()

    # 1) diagonal-adjacent equal pairs, tagged by digit
    for r in range(4):
        for c in range(4):
            v = int(grid[r, c])
            for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 4 and 0 <= c2 < 4 and int(grid[r2, c2]) == v:
                    a, b = (r, c), (r2, c2)
                    a, b = tuple(sorted((a, b)))
                    s.add(("diag", v, a, b))

    # 2) X motifs in each 2x2 block, keep diagonal roles (DO NOT sort digits)
    for br in (0, 2):
        for bc in (0, 2):
            main1 = int(grid[br, bc])
            main2 = int(grid[br+1, bc+1])
            anti1 = int(grid[br, bc+1])
            anti2 = int(grid[br+1, bc])

            if main1 == main2 and anti1 == anti2:
                s.add(("xblock", (br, bc), main1, anti1))

    return s

def is_hidden(inner, outer):
    return motif_set(inner).issubset(motif_set(outer))

def classify_reps_by_hidden(reps):
    n = len(reps)

    contains = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                contains[i][j] = is_hidden(reps[j], reps[i])  # j hidden in i

    containers = [i for i in range(n) if not any(contains[k][i] for k in range(n) if k != i)]

    groups = {i: [i] for i in containers}
    for j in range(n):
        if j in containers:
            continue
        # choose smallest container that contains j (tightest)
        candidates = [i for i in containers if contains[i][j]]
        if candidates:
            best = min(candidates, key=lambda i: len(motif_set(reps[i])))
            groups[best].append(j)

    return groups

def classify_all_into_containers(sols, reps, hidden_groups):
    """
    Returns:
      assign: dict[container_id or None] -> list[np.ndarray]
      chosen: list[container_id or None] aligned with sols indices
    """
    container_ids = sorted(hidden_groups.keys())

    sol_motifs = [motif_set(g) for g in sols]
    cont_motifs = {cid: motif_set(reps[cid]) for cid in container_ids}

    container_ids = sorted(container_ids, key=lambda cid: len(cont_motifs[cid]))

    assign = defaultdict(list)
    chosen = [None] * len(sols)

    for i, (g, mg) in enumerate(zip(sols, sol_motifs)):
        picked = None
        for cid in container_ids:
            if mg.issubset(cont_motifs[cid]):
                picked = cid
                break
        chosen[i] = picked
        assign[picked].append(g)

    return dict(assign), chosen

def expand_containers_until_full_cover(sols, reps, hidden_groups):
    """
    Returns:
      final_containers : list[np.ndarray]  (actual grids, not indices)
      assign           : final assignment dict
    """
    # start from existing containers (actual grids)
    containers = [reps[cid] for cid in sorted(hidden_groups.keys())]

    while True:
        # build temporary reps list
        temp_reps = containers

        # fake hidden_groups: every rep is a container
        temp_hidden = {i: [i] for i in range(len(temp_reps))}

        assign, chosen = classify_all_into_containers(sols, temp_reps, temp_hidden)

        unassigned = assign.get(None, [])
        if not unassigned:
            break  # full cover achieved

        # promote one unassigned sudoku to container
        new_container = unassigned[0]
        containers.append(new_container)

    return containers, assign
