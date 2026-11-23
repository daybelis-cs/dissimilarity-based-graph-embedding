"""
Extract all frequent/maximal/closed patterns from a LAGP or GAGP pattern set.
Builds the embeddings in the corresponding fold and saves
the resulting matrices, pattern properties, and timing logs.

The script runs on ONE sub-base per execution (e.g. Coenen 100/200/300 or
PTC FM/FR/MM/MR). To process several sub-bases, call the script once per
sub-base.
"""

import os
import time
import re 

import numpy as np
import networkx as nx

from utils import loadPatters, openLg, evaluatePatternOnGraph


# =============================================================================
# Configuration (fixed for this example script)
# =============================================================================
# DB determines both the sub-base naming and the class labels:
#   "Coenen" -> sub-bases: 100, 200, 300  ;  labels: 0, 1
#   "PTC"    -> sub-bases: FM, FR, MM, MR      ;  labels: -1, 1
DB        = "Coenen"      # "PTC" or "Coenen"
SUB_BASE  = "200"       # 100,200,300 for Coenen ;  FM, FR, MM, MR for PTC
APPROACH  = "GAGP"     # "LAGP" or "GAGP"
FOLD      = 2          # 1 to 5 
SUPPORT   = "90"      # 80,90 for GAGP; 1,2 for LAGP
DISSIM    = 1		# 1,2

LABELS_BY_DB = {
    "Coenen": [0, 1],
    "PTC":    [-1, 1],
}
LABELS = LABELS_BY_DB[DB]

OUTPUT_DIR = "Output"


# =============================================================================
# Path builders
# =============================================================================
def _fold_dir():
    return os.path.join("FOLDS", DB, str(FOLD))


def _pattern_dir():
    return os.path.join("Patterns", APPROACH, DB, str(FOLD))


def training_path(label):
    return os.path.join(
        _fold_dir(), f"{DB}_{SUB_BASE}_{label}_training.lg"
    )


def evaluation_path(label):
    return os.path.join(
        _fold_dir(), f"{DB}_{SUB_BASE}_{label}_evaluation.lg"
    )


def pattern_path(label):
    return os.path.join(
        _pattern_dir(),
        f"P_{SUB_BASE}_{label}_sup_{SUPPORT}_dis_{DISSIM}.log",
    )


def file_prefix():
    return f"{DB}_{SUB_BASE}_fold{FOLD}_dis_{DISSIM}_support{SUPPORT}"


# =============================================================================
# Evaluate Configurations
# =============================================================================

def _natural_sort(values):
    """Sort ['2','3','80','100'] numerically, and words alphabetically."""
    def key(v):
        try:
            return (0, int(v))
        except ValueError:
            return (1, v)
    return sorted(values, key=key)


def _scan_available_params(sub_base, label):
    """Return (supports, dissims) actually present on disk for this sub_base/label."""
    pdir = _pattern_dir()
    if not os.path.isdir(pdir):
        return set(), set()
    prefix = f"P_{sub_base}_{label}_sup_"
    supports, dissims = set(), set()
    for name in os.listdir(pdir):
        if not name.startswith(prefix):
            continue
        m = re.match(r"(.+?)_dis_(.+)\.log$", name[len(prefix):])
        if m:
            supports.add(m.group(1))
            dissims.add(m.group(2))
    return supports, dissims


def validate_inputs():
    """Return a list of human-readable errors. Empty list means inputs are OK."""
    errors = []

    fold_dir = _fold_dir()
    if not os.path.isdir(fold_dir):
        errors.append(f"Fold directory not found: {fold_dir}")

    for label in LABELS:
        for path, kind in [
            (training_path(label),   "training"),
            (evaluation_path(label), "evaluation"),
        ]:
            if not os.path.exists(path):
                errors.append(f"{kind} graph file not found: {path}")

    pdir = _pattern_dir()
    if not os.path.isdir(pdir):
        errors.append(f"Pattern directory not found: {pdir}")
        return errors

    for label in LABELS:
        expected = pattern_path(label)
        if os.path.exists(expected):
            continue

        supports, dissims = _scan_available_params(SUB_BASE, label)

        if not supports and not dissims:
            errors.append(
                f"No pattern files at all for sub_base={SUB_BASE!r}, "
                f"label={label} in {pdir}"
            )
            continue

        if SUPPORT not in supports:
            errors.append(
                f"SUPPORT={SUPPORT!r} not available for sub_base={SUB_BASE!r}, "
                f"label={label}.  Available supports: "
                f"{_natural_sort(supports)}"
            )
            continue

        if str(DISSIM) not in dissims:
            errors.append(
                f"DISSIM={DISSIM!r} not available for sub_base={SUB_BASE!r}, "
                f"label={label}.  Available dissims: "
                f"{_natural_sort(dissims)}"
            )
            continue

        # support and dissim each exist separately, but this pair does not
        errors.append(
            f"Combination SUPPORT={SUPPORT!r} + DISSIM={DISSIM!r} not available "
            f"for sub_base={SUB_BASE!r}, label={label}.  "
            f"Available supports: {_natural_sort(supports)};  "
            f"available dissims: {_natural_sort(dissims)}"
        )

    return errors
# =============================================================================
# Subgraph / maximal / closed 
# =============================================================================
def is_subgraph(p1, p2, max_dis):
    """Return True if p1 is a subgraph of p2 within max_dis."""
    if len(p1.nodes()) > len(p2.nodes()) or len(p1.edges()) > len(p2.edges()):
        return False
    return evaluatePatternOnGraph(p1, p2, max_dis) <= max_dis


def compute_maximal_closed(pattern, support, patterns, supports):
    """Return (is_closed, is_maximal) for `pattern`."""
    is_maximal = True
    is_closed  = True
    for j in range(len(patterns)):
        if pattern != patterns[j] and is_subgraph(pattern, patterns[j], 0):
            is_maximal = False
            if supports[j] >= support:
                is_closed = False
    return is_closed, is_maximal


# =============================================================================
# Output directory tree
# =============================================================================
def build_output_tree(base_dir):
    subdirs = {
        "all":    os.path.join(base_dir, "matrices_all"),
        "max":    os.path.join(base_dir, "matrices_maximal"),
        "closed": os.path.join(base_dir, "matrices_closed"),
        "pattern_properties":  os.path.join(base_dir, "pattern_properties"),
        "times":  os.path.join(base_dir, "embedding_times"),
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    return subdirs


# =============================================================================
# Main
# =============================================================================
def main():
    out = build_output_tree(OUTPUT_DIR)
    prefix = file_prefix()

    # ------------------------------------------------------------------ 
    # 1) Load patterns and split into all / maximal / closed            
    # ------------------------------------------------------------------ 
    print(f"Loading patterns for {DB} / {SUB_BASE} / {APPROACH} ...")
    patterns_all, supports_all = [], []
    patterns_max, patterns_closed = [], []
    wl_all, wl_max, wl_closed = [], [], []

    t0 = time.time()
    for label in LABELS:
        patterns, supports = loadPatters(
            training_path(label), pattern_path(label)
        )

        for i in range(len(patterns)):
            is_closed, is_maximal = compute_maximal_closed(
                patterns[i], supports[i], patterns, supports
            )

            wl = nx.weisfeiler_lehman_graph_hash(
                patterns[i], edge_attr="label", node_attr="label"
            )

            if wl not in wl_all:
                wl_all.append(wl)
                patterns_all.append(patterns[i])
                supports_all.append(supports[i])

            if is_closed and wl not in wl_closed:
                wl_closed.append(wl)
                patterns_closed.append(patterns[i])

            if is_maximal and wl not in wl_max:
                wl_max.append(wl)
                patterns_max.append(patterns[i])

    elapsed = time.time() - t0
    print(
        f"All: {len(wl_all)}  Closed: {len(patterns_closed)}  "
        f"Maximal: {len(patterns_max)}"
    )

    # ------------------------------------------------------------------ 
    # 2) Pattern properties file                                                
    # ------------------------------------------------------------------ 
    pattern_properties_file = os.path.join(out["pattern_properties"], f"{prefix}_pattern_properties.log")
    with open(pattern_properties_file, "w") as f:
        f.write(f"Patterns {len(patterns_all)}\n")
        f.write(f"Closed   {len(patterns_closed)}\n")
        f.write(f"Maximal  {len(patterns_max)}\n")
        f.write(f"TIME     {elapsed}\n")
        f.write("WL\tClosed\tMaximal\tSupport\tNodes\tEdges\n")
        for i, wl in enumerate(wl_all):
            f.write(
                f"{wl}\t{wl in wl_closed}\t{wl in wl_max}\t"
                f"{supports_all[i]}\t"
                f"{len(patterns_all[i].nodes())}\t"
                f"{len(patterns_all[i].edges())}\n"
            )

    # ------------------------------------------------------------------ 
    # 3) Load graphs                                                    
    # ------------------------------------------------------------------ 
    print("Loading graphs ...")
    train_graphs, train_labels = [], []
    eval_graphs,  eval_labels  = [], []

    for label in LABELS:
        for g in openLg(training_path(label)):
            train_graphs.append(g)
            train_labels.append(label)
        for g in openLg(evaluation_path(label)):
            eval_graphs.append(g)
            eval_labels.append(label)

    # ------------------------------------------------------------------ 
    # 4) Embedding                                                       
    # ------------------------------------------------------------------ 
    def compute_matrices(graphs, tag):
        m_all    = np.zeros((len(graphs), len(patterns_all)))
        m_max    = np.zeros((len(graphs), len(patterns_max)))
        m_closed = np.zeros((len(graphs), len(patterns_closed)))
        t_tot = t_max = t_closed = 0.0

        for gi, g in enumerate(graphs):
            print(f"  [{tag}] graph {gi + 1}/{len(graphs)}")
            for pi, p in enumerate(patterns_all):
                start = time.time()
                dist = nx.graph_edit_distance(p, g)
                if dist == np.inf:
                    dist = len(g.nodes()) + len(g.edges())
                m_all[gi, pi] = dist
                dt = time.time() - start
                t_tot += dt

                wl = wl_all[pi]
                if wl in wl_max:
                    m_max[gi, wl_max.index(wl)] = dist
                    t_max += dt
                if wl in wl_closed:
                    m_closed[gi, wl_closed.index(wl)] = dist
                    t_closed += dt

        return m_all, m_max, m_closed, (t_tot, t_max, t_closed)

    print("Building training matrices ...")
    tr_all, tr_max, tr_closed, tr_times = compute_matrices(train_graphs, "train")

    print("Building evaluation matrices ...")
    ev_all, ev_max, ev_closed, ev_times = compute_matrices(eval_graphs, "eval")

    # ------------------------------------------------------------------ 
    # 5) Save matrices                                               
    # ------------------------------------------------------------------ 
    def save(folder_key, split_name, mat, gt):
        np.save(os.path.join(out[folder_key], f"{prefix}_{split_name}_mat.npy"),  mat)
        np.save(os.path.join(out[folder_key], f"{prefix}_{split_name}_GT.npy"),   gt)

    save("all",    "training",   tr_all,    train_labels)
    save("max",    "training",   tr_max,    train_labels)
    save("closed", "training",   tr_closed, train_labels)
    save("all",    "evaluation", ev_all,    eval_labels)
    save("max",    "evaluation", ev_max,    eval_labels)
    save("closed", "evaluation", ev_closed, eval_labels)

    times_file = os.path.join(out["times"], f"{prefix}_times.txt")
    with open(times_file, "w") as f:
        f.write(f"T_TRAINING   "
                f"T_TOT={tr_times[0]}  T_MAX={tr_times[1]}  T_CER={tr_times[2]}\n")
        f.write(f"T_EVALUATION "
                f"T_TOT={ev_times[0]}  T_MAX={ev_times[1]}  T_CER={ev_times[2]}\n")

    print("[OK]")


if __name__ == "__main__":
    errors = validate_inputs()
    if errors:
        print("\n[ERROR] Invalid configuration — cannot run:\n")
        for e in errors:
            print(f"  - {e}")
        print(
            f"\nCurrent settings: DB={DB!r}  SUB_BASE={SUB_BASE!r}  "
            f"APPROACH={APPROACH!r}  FOLD={FOLD}  "
            f"SUPPORT={SUPPORT!r}  DISSIM={DISSIM!r}"
        )
        raise SystemExit(1)

    out = build_output_tree(OUTPUT_DIR)
    prefix = file_prefix()
    main()
