"""Connectome representation, real-data loaders, and a synthetic stand-in.

WHAT A CONNECTOME IS AND IS NOT
-------------------------------
MaleCNS v1.0 gives you, per neuron: a 3D shape, a cell type, a side, a predicted
neurotransmitter, and a *synapse count* to every other neuron.  It does not give
you conductances, time constants, or release probabilities.  Those do not exist
in the dataset and cannot be measured by EM.  Any "fly brain plays X" project is
therefore making two guesses:

    1. weight_ij  ~ f(synapse_count_ij)          <- you choose f
    2. per-neuron dynamics                        <- you choose the model

Everything downstream (including anything that looks like learning) is
dominated by those two choices unless you keep the learned parameters in the
readout, which is what this project does.  Being explicit about this is the
difference between a demo and an experiment.

This module provides:
    Connectome          sparse, typed graph; CSR-ish lists; stdlib-only
    synthetic_...()     a statistically plausible stand-in so the pipeline runs
    load_malecns_flat() real flat-connectome feather/tables (needs pandas)
    load_neuprint()     live queries against neuPrint (needs neuprint-python)
    flight_subgraph()   prune a graph down to a usable flight-control circuit
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Cell classes.  Names follow the MaleCNS / FlyWire annotation vocabulary.
# The SENSORY -> ... -> DESCENDING spine is the pathway this project rides:
#   photoreceptors -> lamina -> medulla -> T4/T5 motion -> VPN -> DN
FLIGHT_VISUAL_PATHWAY = [
    ("R1-R6", 0),      # photoreceptors, brightness (motion input)
    ("R7", 0), ("R8", 0),   # colour
    ("L1-L5", 1), ("Mi1", 1), ("Tm3", 1),          # lamina / medulla
    ("T4", 2), ("T5", 2),                          # direction-selective motion
    ("LC4", 2), ("LPLC2", 2),                      # looming detectors
    ("VS", 3), ("HS", 3),                          # lobula-plate tangential cells
    # STMD: small-target motion detectors (the lobula's target-selective cells).
    # Flies track small moving objects -- prey, conspecifics, a rival aircraft --
    # with a dedicated high-gain, tightly retinotopic pathway rather than with
    # the wide-field optomotor system.  Without it, a 0.4-degree target is one
    # receptor out of ninety-six pooled into a broad dendritic tree, and its
    # bearing is lost by the third synapse.  This pathway is what a pursuit
    # behaviour actually rides on.
    ("STMD", 3),
    ("VPN", 3),                                    # visual projection neurons
    ("DNp26", 4), ("DNp57", 4), ("DNp03", 4),      # saccade / steering hubs
    ("DNg02", 4), ("DNp06", 4), ("DNa02", 4), ("DNg13", 4),
    ("DNp10", 4), ("DNHS1", 4),
]

# Descending-neuron groups we read the aircraft command out of.  Sign of the
# contribution and the axis it drives are documented in brain/readout.py.
DN_GROUPS = {
    "turn_left":  ["DNp26", "DNp57", "DNg02"],
    "turn_right": ["DNp03", "DNa02", "DNg13"],
    "pitch_up":   ["DNp06", "DNg02"],
    "pitch_down": ["DNHS1"],
    "speed":      ["DNp06", "DNg13"],
    "trigger":    ["DNp10", "DNp03"],
}

REWARD_DA = ["PAM", "PPL1-α3", "PPL3"]        # reward / approach (dopamine)
PUNISH_DA = ["PPL1-γ2α'1", "PPL1-γ1", "PPL101"]  # punishment / avoidance
PLASTIC_PRE = ["KC"]                            # mushroom-body Kenyon cells
PLASTIC_POST = ["MBON"]                         # mushroom-body output neurons


@dataclass
class Connectome:
    ids: List[int] = field(default_factory=list)
    types: List[str] = field(default_factory=list)
    sides: List[str] = field(default_factory=list)
    transmitter: List[str] = field(default_factory=list)
    # receptive-field centre (radians) for retinotopic cells; 0 elsewhere.
    # The real dataset does not ship this, so it is derived from optic-lobe
    # soma position or from the column index of the parent ommatidium.
    pref_az: List[float] = field(default_factory=list)
    pref_el: List[float] = field(default_factory=list)
    # sparse edges: pre[i] -> post[i] with weight[i]
    pre: List[int] = field(default_factory=list)
    post: List[int] = field(default_factory=list)
    weight: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_edges(self) -> int:
        return len(self.pre)

    def index_by_type(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for i, t in enumerate(self.types):
            out.setdefault(t, []).append(i)
        return out

    def incoming(self) -> List[List[Tuple[int, float]]]:
        """CSR-style incoming adjacency: for each neuron, [(pre_idx, w)]."""
        inc: List[List[Tuple[int, float]]] = [[] for _ in range(len(self.ids))]
        for a, b, w in zip(self.pre, self.post, self.weight):
            inc[b].append((a, w))
        return inc

    def outgoing(self) -> List[List[Tuple[int, float]]]:
        out: List[List[Tuple[int, float]]] = [[] for _ in range(len(self.ids))]
        for a, b, w in zip(self.pre, self.post, self.weight):
            out[a].append((b, w))
        return out

    def select(self, indices: Sequence[int]) -> "Connectome":
        """Induced subgraph on `indices` (keeps only edges inside the set)."""
        keep = {old: new for new, old in enumerate(indices)}
        c = Connectome()
        for old in indices:
            c.ids.append(self.ids[old])
            c.types.append(self.types[old])
            c.sides.append(self.sides[old])
            c.transmitter.append(self.transmitter[old])
            c.pref_az.append(self.pref_az[old] if old < len(self.pref_az) else 0.0)
            c.pref_el.append(self.pref_el[old] if old < len(self.pref_el) else 0.0)
        for a, b, w in zip(self.pre, self.post, self.weight):
            if a in keep and b in keep:
                c.pre.append(keep[a])
                c.post.append(keep[b])
                c.weight.append(w)
        return c

    def stats(self) -> dict:
        deg_out = [0] * len(self.ids)
        deg_in = [0] * len(self.ids)
        for a, b in zip(self.pre, self.post):
            deg_out[a] += 1
            deg_in[b] += 1
        types = {}
        for t in self.types:
            types[t] = types.get(t, 0) + 1
        return {"neurons": len(self.ids), "edges": self.n_edges,
                "mean_out_degree": sum(deg_out) / max(len(deg_out), 1),
                "mean_in_degree": sum(deg_in) / max(len(deg_in), 1),
                "types": types}


# --------------------------------------------------------------------------
def synthetic_connectome(scale: float = 1.0, seed: int = 0,
                         neurons: Optional[int] = None) -> Connectome:
    """A statistically plausible stand-in for the real connectome.

    Purpose: let the whole pipeline (sensors -> spiking net -> DN readout ->
    reward -> plasticity) be developed and unit-tested without a 1.1 GB
    download and without a GPU, and let `--brain synthetic` be a documented
    control condition against `--brain malecns`.

    It is NOT a model of the fly.  What it does reproduce, because those are
    the properties the control loop is sensitive to:
      * the layered visual pathway above, with realistic per-layer sizes,
      * log-normal-ish degree distributions (a few hubs, many sparse cells),
      * separate ON/OFF and L/R channels (so antisymmetric steering is possible),
      * dedicated dopamine populations and a plastic mushroom-body-like layer.
    """
    rng = random.Random(seed)
    c = Connectome()
    layer_of: List[int] = []
    type_of: List[str] = []

    def add(t: str, n: int, layer: int, side: str = ""):
        for k in range(n):
            c.ids.append(len(c.ids) + 1000)
            c.types.append(t)
            c.sides.append(side or ("L" if k % 2 == 0 else "R"))
            c.transmitter.append("ACH" if rng.random() < 0.8 else "GABA")
            layer_of.append(layer)
            type_of.append(t)

    base = {
        "R1-R6": 96, "R7": 12, "R8": 12, "L1-L5": 48, "Mi1": 24, "Tm3": 24,
        "T4": 32, "T5": 32, "LC4": 8, "LPLC2": 8, "VS": 16, "HS": 16, "VPN": 24,
        "DNp26": 3, "DNp57": 3, "DNp03": 3, "DNg02": 3, "DNp06": 3,
        "DNa02": 3, "DNg13": 3, "DNp10": 3, "DNHS1": 3,
        "STMD": 24,
        "KC": 200, "MBON": 24, "PAM": 12,
        "PPL1-γ2α'1": 6, "PPL1-γ1": 6, "PPL101": 2,
    }
    for name, n in base.items():
        add(name, max(1, int(round(n * scale))), _layer_of_name(name))
    # retina: one canonical grid.  Fly eyes wrap ~360 deg in azimuth and
    # ~160 deg in elevation, and ommatidial angular spacing is ~4.5 deg.
    c.pref_az = [0.0] * len(c.ids)
    c.pref_el = [0.0] * len(c.ids)
    for k, i in enumerate(c.index_by_type()["R1-R6"]):
        col = k % 12
        row = k // 12
        c.pref_az[i] = math.pi * (2.0 * (col + 0.5) / 12.0 - 1.0)
        c.pref_el[i] = math.radians(-80.0 + 160.0 * (row + 0.5) / 8.0)

    # ---- RETINOTOPY -------------------------------------------------------
    # The single most important structural property for this project: a target
    # at a bearing has to produce a *differential* drive between left and right
    # descending neurons, or the readout has nothing to steer with.  A random
    # edge-generating scheme produces a network whose DN rates are identical
    # whatever the target does, and no amount of learning fixes that.
    by_type = c.index_by_type()
    def near(i, j, tol_az_deg, tol_el_deg=None):
        tol_el_deg = tol_el_deg if tol_el_deg is not None else tol_az_deg
        daz = abs(_wrap_pi(c.pref_az[i] - c.pref_az[j]))
        del_ = abs(c.pref_el[i] - c.pref_el[j])
        return daz < math.radians(tol_az_deg) and del_ < math.radians(tol_el_deg)

    def inherit(types, tol_az_deg, w_mu, parent="R1-R6"):
        """Wire each post cell to neighbours on the retina (tight kernel).

        Tolerances are in degrees of visual angle and are deliberately small:
        a 47-degree 'nearby' tolerance makes the whole pathway non-retinotopic
        and no amount of downstream learning recovers a bearing signal.
        """
        for t in types:
            for j in by_type.get(t, []):
                src = rng.choice(by_type[parent])
                c.pref_az[j] = c.pref_az[src] + math.radians(rng.gauss(0, 3.0))
                c.pref_el[j] = c.pref_el[src] + math.radians(rng.gauss(0, 4.0))
                cands = [i for i in by_type["R1-R6"] if near(i, j, tol_az_deg)]
                for i in rng.sample(cands, min(len(cands), 4)):
                    c.pre.append(i)
                    c.post.append(j)
                    c.weight.append(abs(rng.gauss(w_mu, w_mu * 0.25)))

    inherit(["L1-L5", "Mi1", "Tm3"], 14.0, 1.3)

    # T4/T5: direction-selective; each keeps a preferred direction so the pair
    # encodes motion, which is what a pursuit loop needs to damp.
    for t in ("T4", "T5"):
        for j in by_type.get(t, []):
            src = rng.choice(by_type["R1-R6"])
            c.pref_az[j] = c.pref_az[src] + math.radians(rng.gauss(0, 3.0))
            c.pref_el[j] = c.pref_el[src] + math.radians(rng.gauss(0, 4.0))
            cands = [i for i in by_type["L1-L5"] + by_type["Mi1"] + by_type["Tm3"]
                     if near(i, j, 22.0)]
            for i in rng.sample(cands, min(len(cands), 4)):
                c.pre.append(i)
                c.post.append(j)
                c.weight.append(abs(rng.gauss(1.5, 0.35)))

    # VS/HS: wide-field tangential cells, one per side, pooling T4/T5 of that side
    for j in by_type["VS"] + by_type["HS"]:
        side = rng.choice(["L", "R"])
        c.sides[j] = side
        c.pref_az[j] = math.radians(-120.0 if side == "L" else 120.0)
        c.pref_el[j] = 0.0
        srcs = [i for i in by_type["T4"] + by_type["T5"]
                if (c.pref_az[i] < 0.0) == (side == "L")]
        for i in rng.sample(srcs, min(len(srcs), 8)):
            c.pre.append(i)
            c.post.append(j)
            c.weight.append(abs(rng.gauss(1.2, 0.3)))

    # looming detectors: wide-field, no retinotopy
    for j in by_type["LC4"] + by_type["LPLC2"]:
        c.pref_az[j] = 0.0
        c.pref_el[j] = 0.0
        for i in rng.sample(by_type["T4"] + by_type["T5"], 4):
            c.pre.append(i)
            c.post.append(j)
            c.weight.append(abs(rng.gauss(1.4, 0.3)))

    # small-target detectors: direct, tight, high-gain from the retina
    for j in by_type["STMD"]:
        src = rng.choice(by_type["R1-R6"])
        c.pref_az[j] = c.pref_az[src] + math.radians(rng.gauss(0, 2.0))
        c.pref_el[j] = c.pref_el[src] + math.radians(rng.gauss(0, 3.0))
        c.sides[j] = "R" if c.pref_az[j] >= 0.0 else "L"
        cands = [i for i in by_type["R1-R6"] if near(i, j, 9.0)]
        for i in rng.sample(cands, min(len(cands), 3)):
            c.pre.append(i); c.post.append(j)
            c.weight.append(abs(rng.gauss(2.6, 0.5)))

    # VPNs: retinotopic readout, explicitly sided by the sign of their bearing
    for j in by_type["VPN"]:
        src = rng.choice(by_type["T4"] + by_type["T5"])
        c.pref_az[j] = c.pref_az[src]
        c.pref_el[j] = c.pref_el[src]
        c.sides[j] = "R" if c.pref_az[j] >= 0.0 else "L"
        for i in rng.sample(by_type["T4"] + by_type["T5"] + by_type["VS"] +
                            by_type["HS"] + by_type["LC4"] + by_type["LPLC2"], 6):
            c.pre.append(i)
            c.post.append(j)
            c.weight.append(abs(rng.gauss(1.3, 0.3)))
        # and the small-target channel, with the same receptive field
        for i in _top4(rng, [k for k in by_type["STMD"]
                             if abs(_wrap_pi(c.pref_az[k] - c.pref_az[j])) < math.radians(20.0)]):
            c.pre.append(i); c.post.append(j); c.weight.append(abs(rng.gauss(2.2, 0.4)))

    # DN layer: LEFT-field VPNs drive the RIGHT-turn group and vice versa, so a
    # target on one side produces an antisymmetric steering drive.  The readout
    # sign is then a property of the brain, not of the reward function.
    turn_left = [i for t in DN_GROUPS["turn_left"] for i in by_type.get(t, [])]
    turn_right = [i for t in DN_GROUPS["turn_right"] for i in by_type.get(t, [])]
    for group, side in ((turn_left, "L"), (turn_right, "R")):
        srcs = [i for i in by_type["VPN"] if c.sides[i] == side]
        stmd = [i for i in by_type["STMD"] if c.sides[i] == side]
        for j in group:
            for i in _top4(rng, srcs):
                c.pre.append(i); c.post.append(j); c.weight.append(abs(rng.gauss(1.6, 0.4)))
            # the pursuit command leans on the small-target channel
            for i in _top4(rng, stmd):
                c.pre.append(i); c.post.append(j); c.weight.append(abs(rng.gauss(2.4, 0.5)))

    def connect(pre_types, post_types, p: float, w_mu: float, max_n: int = 40):
        pres = [i for t in pre_types for i in by_type.get(t, [])]
        posts = [i for t in post_types for i in by_type.get(t, [])]
        for a in pres:
            for b in rng.sample(posts, min(len(posts), _log_normal(rng, max_n))):
                if rng.random() < p:
                    c.pre.append(a)
                    c.post.append(b)
                    c.weight.append(abs(rng.gauss(w_mu, w_mu * 0.3)))

    # vertical channel: upper vs lower visual field drives the pitch groups
    for name, want_up in (("pitch_up", True), ("pitch_down", False)):
        for t in DN_GROUPS[name]:
            for j in by_type.get(t, []):
                srcs = [i for i in by_type["VPN"]
                        if (c.pref_el[i] > 0.0) == want_up]
                for i in _top4(rng, srcs):
                    c.pre.append(i); c.post.append(j); c.weight.append(abs(rng.gauss(1.1, 0.3)))
    # optic-flow / optomotor drive: wide-field, so it must NOT be sprayed over
    # every DN.  A broad VS/HS projection onto the whole descending population
    # adds a large common-mode signal that swamps the few-spike small-target
    # signal carrying the target's bearing (measured: the lateral DN channel
    # then shows a constant offset and no bearing dependence at all).  Keep it
    # to the straight-flight and speed DNs, which is where the optomotor
    # response actually belongs.
    connect(["VS", "HS"], ["DNp06", "DNg13"], 0.30, 0.9)
    # looming ->
    connect(["LC4", "LPLC2"], ["DNHS1", "DNp10"], 0.5, 1.2)
    # recurrent DN network (real one is strongly recurrent; DNb01 is inhibitory
    # between the saccade and straight-flight DNs)
    connect(["DNp26", "DNp57", "DNp03"], ["DNp26", "DNp57", "DNp03", "DNg02",
                                          "DNp06", "DNa02", "DNg13"], 0.35, 0.8)
    # mushroom body: KC <- VPN, MBON <- KC, dopamine gates the KC->MBON edge
    connect(["VPN", "DNp03"], ["KC"], 0.05, 0.8, max_n=12)
    connect(["KC"], ["MBON"], 0.02, 0.7, max_n=16)
    connect(["MBON"], ["DNp03", "DNg02", "DNp06"], 0.4, 0.9)
    connect(["PAM", "PPL1-γ2α'1", "PPL1-γ1", "PPL101"], ["KC", "MBON"], 0.5, 1.0)

    return _apply_transmitter_signs(c)


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _top4(rng, srcs):
    if not srcs:
        return []
    return rng.sample(srcs, min(len(srcs), 4))


def _layer_of_name(t: str) -> int:
    for name, layer in FLIGHT_VISUAL_PATHWAY:
        if t == name:
            return layer
    if t in ("KC", "MBON", "PAM"):
        return 5
    return 5


def _log_normal(rng: random.Random, mu: int) -> int:
    return max(1, int(rng.lognormvariate(math.log(mu), 0.8)))


def _apply_transmitter_signs(c: Connectome) -> Connectome:
    """Make inhibitory transmitters produce negative weights.

    The MaleCNS release includes per-neuron neurotransmitter predictions
    (`body-neurotransmitters-male-cns-v1.0.feather`).  Using them as the sign of
    the connection is the cheapest large win available when building a
    connectome-constrained model: a purely excitatory or sign-random network
    produces runaway synchrony and no steering at all.
    """
    inhib_types = {"GABA", "GLUT", "GLY"}
    for k in range(len(c.pre)):
        pre = c.pre[k]
        if c.transmitter[pre] in inhib_types:
            c.weight[k] = -abs(c.weight[k])
        else:
            c.weight[k] = abs(c.weight[k])
    return c


# --------------------------------------------------------------------------
def flight_subgraph(c: Connectome) -> Connectome:
    """Prune to the visuomotor + steering spine (what a BFM controller needs)."""
    want = set()
    for name, _ in FLIGHT_VISUAL_PATHWAY:
        want.add(name)
    for group in DN_GROUPS.values():
        want.update(group)
    want.update(REWARD_DA)
    want.update(PUNISH_DA)
    want.update(PLASTIC_PRE)
    want.update(PLASTIC_POST)
    idx = [i for i, t in enumerate(c.types) if t in want]
    return c.select(idx)


# --------------------------------------------------------------------------
def load_malecns_flat(connectome_path: str, annotations_path: str,
                      min_confidence: float = 0.5,
                      min_synapses: int = 1,
                      weight_fn=None) -> Connectome:
    """Load the real MaleCNS v1.0 flat connectome.

    Files (CC-BY 4.0, no credentials, ~1.1 GB total) live at
    `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` and are served
    over https as well:

        body-annotations-male-cns-v1.0-minconf-0.5.feather      13 MB
        body-neurotransmitters-male-cns-v1.0.feather            42 MB
        connectome-weights-male-cns-v1.0-minconf-0.5.feather   1.1 GB

    Column names differ between releases, so the loader sniffs for the obvious
    candidates and raises a clear error listing what it found.  Requires
    `pandas` (or `pyarrow`) -- both absent from a bare stdlib environment.

    weight_fn(synapse_count, pre_tx, post_tx) -> float  lets you choose the
    synapse-count-to-weight mapping, which is THE unidentifiable parameter of
    the whole exercise.  Default: w = count**0.5, scaled to [0.05, 3.0].
    """
    try:
        import pandas as pd  # noqa
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "load_malecns_flat needs pandas/pyarrow: pip install pandas pyarrow"
        ) from exc

    ann = pd.read_feather(annotations_path)

    # --- transmitter file: handle both naming variants ---
    # FILES lists body-neurotransmitters-male-cns-v1.0.feather (no minconf)
    # but old code assumed body-neurotransmitters-male-cns-v1.0-minconf-0.5.feather
    # via replace. Try robust detection.
    import os
    tx_path = None
    # 1) try directory scan for any body-neurotransmitters*.feather
    dir_name = os.path.dirname(connectome_path) or "."
    if os.path.isdir(dir_name):
        for fn in os.listdir(dir_name):
            if fn.startswith("body-neurotransmitters") and fn.endswith(".feather"):
                tx_path = os.path.join(dir_name, fn)
                break
    # 2) try replace variants
    if tx_path is None:
        candidates = [
            connectome_path.replace("connectome-weights", "body-neurotransmitters"),
            connectome_path.replace("connectome-weights-male-cns-v1.0-minconf-0.5", "body-neurotransmitters-male-cns-v1.0"),
            connectome_path.replace("connectome-weights", "body-neurotransmitters-male-cns-v1.0"),
            os.path.join(dir_name, "body-neurotransmitters-male-cns-v1.0.feather"),
            os.path.join(dir_name, "body-neurotransmitters-male-cns-v1.0-minconf-0.5.feather"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                tx_path = cand
                break
    # 3) fallback: if still not found, try same dir as annotations
    if tx_path is None:
        ann_dir = os.path.dirname(annotations_path) or "."
        if os.path.isdir(ann_dir):
            for fn in os.listdir(ann_dir):
                if fn.startswith("body-neurotransmitters") and fn.endswith(".feather"):
                    tx_path = os.path.join(ann_dir, fn)
                    break
    if tx_path is None:
        # If transmitter file missing, proceed without it — default ACH
        tx = None
    else:
        tx = pd.read_feather(tx_path)

    # Support JSONL pruned output as well (see fetch_connectome prune)
    if connectome_path.endswith(".jsonl") or annotations_path.endswith(".jsonl"):
        # delegate to jsonl loader
        return load_pruned_jsonl(connectome_path, annotations_path, min_synapses=min_synapses, weight_fn=weight_fn)

    con = pd.read_feather(connectome_path)

    def pick(df, *names):
        for n in names:
            if n in df.columns:
                return n
        raise KeyError(f"none of {names} in columns {list(df.columns)}")

    id_col = pick(ann, "bodyId", "body_id", "body_id", "id", "bodyId_pre", "body_pre")
    type_col = pick(ann, "cell_type", "type", "primary_type", "cellType", "instance_type")
    # Fix annotation schema mismatch: actual dataset has somaSide, rootSide
    side_col = pick(ann, "side", "side_predicted", "somaSide", "rootSide", "soma_side", "root_side", "somaSide_predicted")
    if tx is not None:
        tx_col = pick(tx, "transmitter", "nt_type", "top_nt", "neurotransmitter", "predicted_nt")
        # tx id col may be bodyId or id
        try:
            tx_id_col = pick(tx, "bodyId", "body_id", "id")
        except KeyError:
            tx_id_col = id_col if id_col in tx.columns else list(tx.columns)[0]
    else:
        tx_col = None
        tx_id_col = None
    # Fix edge schema: actual current files use body_pre, body_post, weight
    pre_col = pick(con, "body_pre", "bodyId_pre", "pre", "pre_bodyId", "bodyId_pre")
    post_col = pick(con, "body_post", "bodyId_post", "post", "post_bodyId", "bodyId_post")
    w_col = pick(con, "weight", "syn_count", "count", "synapses", "size")

    ids = list(ann[id_col])
    index = {b: i for i, b in enumerate(ids)}
    c = Connectome(ids=ids, types=[str(t) for t in ann[type_col]],
                   sides=[str(s) for s in ann[side_col]],
                   transmitter=[""] * len(ids))
    if tx is not None and tx_col is not None:
        # tx may have different id column name
        try:
            tx_ids = list(tx[tx_id_col]) if tx_id_col in tx.columns else list(tx[id_col])
        except Exception:
            tx_ids = list(tx.iloc[:,0])
        tx_vals = list(tx[tx_col])
        txmap = dict(zip(tx_ids, tx_vals))
        for i, b in enumerate(ids):
            c.transmitter[i] = str(txmap.get(b, "ACH"))
    else:
        for i in range(len(ids)):
            c.transmitter[i] = "ACH" 

    if weight_fn is None:
        def weight_fn(n, pre_tx, post_tx):
            return min(3.0, max(0.05, float(n) ** 0.5 * 0.15))

    for a, b, n in zip(con[pre_col], con[post_col], con[w_col]):
        if n < min_synapses:
            continue
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            continue
        c.pre.append(ia)
        c.post.append(ib)
        c.weight.append(weight_fn(n, c.transmitter[ia], c.transmitter[ib]))
    return _apply_transmitter_signs(c)


def load_pruned_jsonl(connectome_path: str, annotations_path: str,
                      min_synapses: int = 1,
                      weight_fn=None) -> Connectome:
    """Load the compact JSONL output of `fetch_connectome prune`.

    prune writes:
      flight_subgraph.jsonl  {id, type, side}
      flight_edges.jsonl     {pre, post, w}  where pre/post are bodyIds

    This loader makes that output runnable: build_controller() now reads it
    if FLYBFM_CONNECTOME points to flight_edges.jsonl and FLYBFM_ANNOTATIONS
    to flight_subgraph.jsonl. Previously prune was disconnected from runtime.
    """
    import json
    import os
    # annotations_path should be flight_subgraph.jsonl, connectome_path flight_edges.jsonl
    # but allow either order: detect which file contains 'id' vs 'pre'
    ann_path = annotations_path
    edge_path = connectome_path
    # If user passed them swapped, try to auto-detect
    def sniff(p):
        try:
            with open(p) as fh:
                first = fh.readline()
                if not first:
                    return "unknown"
                obj = json.loads(first)
                if "id" in obj and "type" in obj:
                    return "ann"
                if "pre" in obj and "post" in obj:
                    return "edge"
        except Exception:
            pass
        return "unknown"
    # if edge_path looks like ann and ann_path looks like edge, swap
    if os.path.exists(edge_path) and os.path.exists(ann_path):
        s_edge = sniff(edge_path)
        s_ann = sniff(ann_path)
        if s_edge == "ann" and s_ann == "edge":
            ann_path, edge_path = edge_path, ann_path

    ids = []
    types = []
    sides = []
    id_to_idx = {}
    with open(ann_path) as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bid = int(obj["id"])
            id_to_idx[bid] = len(ids)
            ids.append(bid)
            types.append(str(obj.get("type", "unknown")))
            sides.append(str(obj.get("side", "C")))

    c = Connectome(ids=ids, types=types, sides=sides, transmitter=["ACH"]*len(ids))

    if weight_fn is None:
        def weight_fn(n, pre_tx, post_tx):
            return min(3.0, max(0.05, float(n) ** 0.5 * 0.15))

    with open(edge_path) as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # support both bodyId and idx formats
            pre_raw = obj.get("pre")
            post_raw = obj.get("post")
            w_raw = obj.get("w", obj.get("weight", 1))
            if float(w_raw) < min_synapses:
                continue
            # if pre/post are already indices (small), keep; if bodyIds (large), map
            if pre_raw in id_to_idx and post_raw in id_to_idx:
                ia = id_to_idx[pre_raw]
                ib = id_to_idx[post_raw]
            else:
                # assume they are already indices or direct bodyIds that were mapped
                # try mapping, fallback to raw if within range
                ia = id_to_idx.get(pre_raw)
                ib = id_to_idx.get(post_raw)
                if ia is None or ib is None:
                    # maybe file already uses 0-based indices
                    try:
                        ia_int = int(pre_raw)
                        ib_int = int(post_raw)
                        if 0 <= ia_int < len(ids) and 0 <= ib_int < len(ids):
                            ia, ib = ia_int, ib_int
                        else:
                            continue
                    except Exception:
                        continue
            c.pre.append(ia)
            c.post.append(ib)
            c.weight.append(weight_fn(w_raw, "ACH", "ACH"))
    return _apply_transmitter_signs(c)


def load_neuprint(dataset: str = "male-cns:v1.0", token: str = "",
                  cell_types: Optional[Iterable[str]] = None,
                  server: str = "https://neuprint.janelia.org") -> Connectome:
    """Pull a subgraph live from neuPrint instead of downloading 1.1 GB.

    Get a free token at https://neuprint.janelia.org (account + API token), then
    export NEUPRINT_TOKEN.  Example from the MaleCNS docs:

        from neuprint import Client, fetch_neurons, fetch_adjacencies
        client = Client("https://neuprint.janelia.org", dataset='male-cns:v1.0',
                        token=token)
        neurons, syndist = fetch_neurons("DNge104")
        outgoing, info = fetch_adjacencies("DNge104")
    """
    try:  # pragma: no cover - requires network + credentials
        from neuprint import Client, fetch_neurons, fetch_adjacencies  # noqa
    except Exception as exc:
        raise RuntimeError("pip install neuprint-python, and set NEUPRINT_TOKEN") from exc

    client = Client(server, dataset=dataset, token=token)
    c = Connectome()
    index: Dict[int, int] = {}

    def add(body_id, t, side, tx):
        if body_id not in index:
            index[body_id] = len(c.ids)
            c.ids.append(body_id)
            c.types.append(t)
            c.sides.append(side)
            c.transmitter.append(tx)

    for t in (cell_types or []):
        neurons, _ = fetch_neurons(f"type:{t}")
        bodies = []
        for _, row in neurons.iterrows():
            b = int(row["bodyId"])
            add(b, str(row.get("type", t)), str(row.get("side", "")),
                str(row.get("transmitter", "ACH")))
            bodies.append(b)
        if not bodies:
            continue
        out_edges, _ = fetch_adjacencies(bodies)
        for _, row in out_edges.iterrows():
            a, b, w = int(row["bodyId_pre"]), int(row["bodyId_post"]), float(row["weight"])
            if b in index:
                c.pre.append(index[a])
                c.post.append(index[b])
                c.weight.append(min(3.0, max(0.05, w ** 0.5 * 0.15)))
    return _apply_transmitter_signs(c)
