"""Fetch the MaleCNS v1.0 flat connectome (or print where to get it).

    python -m flybfm.tools.fetch_connectome list
    python -m flybfm.tools.fetch_connectome download --out data/malecns
    python -m flybfm.tools.fetch_connectome prune --out data/malecns

The flat connectome is CC-BY 4.0 and needs no credentials.  It is also ~1.2 GB,
so `--prune` exists: it keeps only the cell types this project needs (visual
projection neurons, small-target and looming cells, descending neurons and their
inputs) and writes a compact edge list the stdlib loader can read without pandas.

If you only want a few neurons, do not download anything -- use neuPrint
(see docs/DATA.md and `brain/connectome.py::load_neuprint`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

BUCKET = "flyem-male-cns"
PREFIX = "v1.0/connectome-data/flat-connectome"
BASE = f"https://storage.googleapis.com/{BUCKET}/{PREFIX}"

FILES = [
    ("body-annotations-male-cns-v1.0-minconf-0.5.feather", "13 MB",
     "cell types, sides, neuropil annotations"),
    ("body-neurotransmitters-male-cns-v1.0.feather", "42 MB",
     "transmitter identity per neuron (sets the sign of each edge)"),
    ("connectome-weights-male-cns-v1.0-minconf-0.5.feather", "1.1 GB",
     "the edges: pre, post, synapse counts, neuropils"),
]

#: Cell types worth keeping for a visual -> descending flight subgraph.
#: Tight flight-relevant default (~1k-5k neurons) — matches neuprint default slice.
#: Previously this was very broad (DNp, DNg, VPN, VLP etc) which kept 24k neurons,
#: far larger than the documented 1k-5k flight slice. Now default is specific.
DEFAULT_KEEP = (
    "R1-R6", "R1-6", "T4", "T5", "LC4", "LPLC2", "STMD", "VPN",
    "VS", "HS",
    "DNp26", "DNp57", "DNp03", "DNg02", "DNp06", "DNa02", "DNg13", "DNp10", "DNHS1",
    "KC", "MBON", "PAM", "PPL101",
)

# Broader set for users who explicitly want more (use --keep with custom list or --keep broad)
BROAD_KEEP = (
    "R1-6", "R7", "R8", "R1-R6", "PHOTORECEPTOR", "T4", "T5", "LPLC2", "LPLC4", "LC4",
    "STMD", "VS", "HS", "VCH", "DCH",
    "DNp", "DNg", "DNa", "DNb", "DNc", "DNd", "DNe", "DNv", "DNx", "DNHS", "MDN",
    "VPN", "TVC", "VLP", "VLPp",
)


def cmd_list(args) -> int:
    print(f"MaleCNS v1.0 flat connectome (CC-BY 4.0)\n{BASE}/\n")
    for name, size, what in FILES:
        print(f"  {name}\n      {size:>7}  {what}")
    print("\nBrowse it interactively instead: https://neuprint.janelia.org  (dataset male-cns:v1.0)")
    print("Cell types: https://reiserlab.github.io/celltype-explorer-drosophila-male-cns/")
    return 0


def cmd_download(args) -> int:
    import urllib.request
    os.makedirs(args.out, exist_ok=True)
    for name, size, _what in FILES:
        dest = os.path.join(args.out, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"have {name} ({os.path.getsize(dest)/1e6:.0f} MB)")
            continue
        if args.skip_weights and "connectome-weights" in name:
            print(f"skipping {name} (--skip-weights)")
            continue
        url = f"{BASE}/{name}"
        print(f"GET {url}  ({size})")
        try:
            with urllib.request.urlopen(url) as resp, open(dest + ".part", "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {100*done/total:5.1f}%  {done/1e6:7.0f}/{total/1e6:.0f} MB",
                              end="", flush=True)
            os.replace(dest + ".part", dest)
            print()
        except Exception as exc:
            print(f"\n  failed: {exc}\n"
                  f"  try: gsutil cp -r gs://{BUCKET}/{PREFIX} {args.out}/")
            return 1
    print(f"\nwrote to {args.out}\nnow: export FLYBFM_CONNECTOME={args.out}/connectome-weights-*.feather")
    print(f"     export FLYBFM_ANNOTATIONS={args.out}/body-annotations-*.feather")
    return 0


def cmd_prune(args) -> int:
    """Reduce the flat connectome to a compact JSONL the stdlib loader can read.

    Output is now wired to runtime: build_controller() can load the JSONL directly
    if FLYBFM_CONNECTOME points to flight_edges.jsonl and FLYBFM_ANNOTATIONS to
    flight_subgraph.jsonl, or it can still load the original feather files.
    """
    try:
        import pandas as pd
    except Exception:
        print("--prune needs pandas + pyarrow to read the .feather files:\n"
              "    pip install pandas pyarrow\n"
              "(the rest of the repo deliberately has no dependencies)", file=sys.stderr)
        return 2

    # Ensure output dir exists — fixes FileNotFoundError when data/malecns_pruned missing
    os.makedirs(args.out, exist_ok=True)

    src = args.src or _find(args.out, "connectome-weights")
    # if not found in out, also look in parent or args.out's sibling data/malecns
    if not src:
        # try data/malecns as fallback if out is pruned dir
        for cand_dir in [args.out, os.path.join(os.path.dirname(args.out), "malecns"), "data/malecns"]:
            if os.path.isdir(cand_dir):
                f = _find(cand_dir, "connectome-weights")
                if f:
                    src = f
                    break
    ann = args.annotations or _find(args.out, "body-annotations")
    if not ann:
        for cand_dir in [args.out, os.path.join(os.path.dirname(args.out), "malecns"), "data/malecns"]:
            if os.path.isdir(cand_dir):
                f = _find(cand_dir, "body-annotations")
                if f:
                    ann = f
                    break
    if not src or not ann:
        print("could not find the feather files; pass --src and --annotations", file=sys.stderr)
        return 2
    print(f"edges : {src}\nannots: {ann}")
    a = pd.read_feather(ann)
    cols = {c.lower(): c for c in a.columns}
    # id, type, side — support many naming variants observed in releases
    id_col = _pick(cols, "bodyid", "body_id", "root_id", "id")
    type_col = _pick(cols, "celltype", "cell_type", "type", "primary_type", "annotation")
    side_col = _pick(cols, "side", "side_predicted", "somaside", "soma_side", "rootside", "root_side", "hemisphere")
    if id_col is None or type_col is None:
        print(f"unexpected annotation columns: {list(a.columns)}", file=sys.stderr)
        return 2
    if args.keep:
        if args.keep.strip().lower() == "broad":
            keep = set(BROAD_KEEP)
        else:
            keep = set(k.strip() for k in args.keep.split(",") if k.strip())
    else:
        keep = set(DEFAULT_KEEP)

    def wanted(t) -> bool:
        # More precise than substring: exact or prefix match.
        # Previously used `k in t` which caused 24k matches (e.g. 'T4' in 'STMD' etc).
        # Now: t == k or t.startswith(k) or t.startswith(k+'_') or t.startswith(k+'-')
        # For DN families like 'DNp', we want prefix match.
        t = str(t).strip()
        if not t:
            return False
        for k in keep:
            k = k.strip()
            if not k:
                continue
            if t == k:
                return True
            if t.startswith(k):
                return True
            # handle case like R1-6 vs R1-R6 etc — allow k without hyphen to match?
            # keep original broad fallback only for very short keys that are known families
            # but avoid overly broad substring.
        return False

    a = a[[c for c in (id_col, type_col, side_col) if c]].rename(
        columns={id_col: "id", type_col: "type", side_col: "side"} if side_col else {id_col: "id", type_col: "type"})
    if "side" not in a.columns:
        a["side"] = "C"
    # filter
    mask = a["type"].map(wanted)
    a_filtered = a[mask]
    ids = set(a_filtered["id"].tolist())
    print(f"kept {len(a_filtered)} annotated neurons of {len(ids)} ids by type (from {len(a)} total)")
    if len(a_filtered) > 8000:
        print(f"WARNING: kept {len(a_filtered)} neurons — much larger than expected 1k-5k flight slice. "
              f"Consider --keep with tighter list e.g. R1-R6,T4,T5,LC4,LPLC2,STMD,VPN,DNp26,DNp57,DNp03,DNg02,DNp06,DNa02,DNg13,DNp10,DNHS1,KC,MBON,PAM,PPL101")
    a = a_filtered

    out = os.path.join(args.out, "flight_subgraph.jsonl")
    with open(out, "w") as fh:
        for row in a.itertuples(index=False):
            side_val = str(getattr(row, "side", "C")) if hasattr(row, "side") else "C"
            fh.write(json.dumps({"id": int(row.id), "type": str(row.type),
                                 "side": side_val}) + "\n")
    print(f"wrote {out}")

    e = pd.read_feather(src, columns=None)
    ecols = {c.lower(): c for c in e.columns}
    # Support observed variants: body_pre/body_post (current), bodyId_pre, pre, etc.
    pre = _pick(ecols, "body_pre", "bodyid_pre", "pre", "pre_bodyid", "bodyid_pre", "from", "source")
    post = _pick(ecols, "body_post", "bodyid_post", "post", "post_bodyid", "bodyid_post", "to", "target")
    w = _pick(ecols, "weight", "synapses", "count", "size", "syn_count")
    if None in (pre, post, w):
        print(f"unexpected edge columns: {list(e.columns)}", file=sys.stderr)
        return 2
    e = e[[pre, post, w]].rename(columns={pre: "pre", post: "post", w: "w"})
    e = e[e["pre"].isin(ids) & e["post"].isin(ids)]
    eout = os.path.join(args.out, "flight_edges.jsonl")
    with open(eout, "w") as fh:
        for row in e.itertuples(index=False):
            fh.write(json.dumps({"pre": int(row.pre), "post": int(row.post),
                                 "w": float(row.w)}) + "\n")
    print(f"wrote {eout}  ({len(e)} edges)\n"
          f"Now you can either:\n"
          f"  1) Use JSONL directly (wired to runtime):\n"
          f"     export FLYBFM_CONNECTOME={eout}\n"
          f"     export FLYBFM_ANNOTATIONS={out}\n"
          f"  2) Or use original feather files with --brain malecns:\n"
          f"     export FLYBFM_CONNECTOME={src}\n"
          f"     export FLYBFM_ANNOTATIONS={ann}")
    return 0


def _pick(cols: Dict[str, str], *names) -> Optional[str]:
    for n in names:
        if n in cols:
            return cols[n]
    return None


def _find(directory: str, needle: str) -> Optional[str]:
    for fn in sorted(os.listdir(directory)):
        if needle in fn and fn.endswith(".feather"):
            return os.path.join(directory, fn)
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fetch_connectome", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list", help="print the file list and sizes")
    l.set_defaults(func=cmd_list, out="data/malecns")
    d = sub.add_parser("download", help="download the flat connectome")
    d.add_argument("--out", default="data/malecns")
    d.add_argument("--skip-weights", action="store_true")
    d.set_defaults(func=cmd_download)
    pr = sub.add_parser("prune", help="reduce to a flight subgraph (needs pandas)")
    pr.add_argument("--out", default="data/malecns")
    pr.add_argument("--src", default=None)
    pr.add_argument("--annotations", default=None)
    pr.add_argument("--keep", default=None)
    pr.set_defaults(func=cmd_prune)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
