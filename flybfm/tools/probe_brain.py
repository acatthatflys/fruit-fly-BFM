"""Open-loop probe of the connectome brain.

    python -m flybfm probe --steps 250 --seed 2

It exists because a brain that is silent, or that is excited equally by every
bearing, looks exactly like a brain that is thinking.  The probe parks the target
at a known angle off the shooter's nose, runs the reduced visual connectome in
open loop, and reports what the retina, the small-target cells and the descending
neurons actually do.

The failure modes it was written to catch, in the order they appear:

1. the whole population is silent        -> bg_current / gain_photo too low;
2. the retina does not track bearing     -> the target's image is smaller than
                                            the receptor spacing, or the mosaic
                                            is not laid out as its own receptive
                                            fields;
3. the small-target cells do not order   -> the visual pathway is not wired
                                            through the lobula plate stage;
4. the descending pair is flat           -> the left and right copies of one
                                            descending type are being summed,
                                            which cancels the lateral signal.

See docs/EXPERIMENTS.md section 3 for the numbers this produced.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import List, Optional

from ..brain.connectome import DN_GROUPS
from ..brain.controller import build_controller
from ..sim.arena import Scenario
from ..sim.dogfight import Dogfight, EnvConfig

VISUAL_TYPES = {"STMD", "T4", "T5", "T2", "T3"}


def _sweep(bearings: List[float], steps: int, seed: int, distance: float,
           side: str = "blue", use_torch: bool = False, torch_device: str | None = None,
           plasticity: bool = False, brain_kind: str = "synthetic") -> List[dict]:
    """One row per commanded angle off the nose, measured geometry reported back."""
    rows: List[dict] = []
    for want in bearings:
        # A fresh environment per bearing: nothing carries over, so a difference
        # between rows is the bearing and nothing else.  `nose_offset_deg` is the
        # shooter's heading relative to the target's, so it moves the target to
        # the *left* of the nose for positive values -- hence the sign flip.
        env = Dogfight(EnvConfig())
        env.reset(Scenario(range_m=distance, taa_deg=0.0,
                           nose_offset_deg=-want, alt_m=6000.0, seed=seed))
        ctl = build_controller(kind=brain_kind, seed=seed, use_torch=use_torch,
                               torch_device=torch_device, plasticity=plasticity)
        conn = ctl.conn
        me = env.blue if side == "blue" else env.red
        other = env.red if side == "blue" else env.blue

        counts: Counter = Counter()
        for _ in range(steps):
            ctl.net.clear_inputs()
            ctl.sensors.inject(ctl.net, me.s, other.s, me.geom_to_other,
                               getattr(me, "los_rate_rad", 0.0), ctl.net.p.dt)
            for i in ctl.net.step():
                counts[i] += 1

        retina = [i for i, t in enumerate(conn.types) if t.startswith("R") and "-" in t]
        visual = [i for i, t in enumerate(conn.types) if t in VISUAL_TYPES]

        def peak(idx: List[int]):
            best, best_n = None, -1
            for i in idx:
                if counts.get(i, 0) > best_n:
                    best, best_n = i, counts.get(i, 0)
            if best is None or best_n <= 0:
                return None, None, 0
            return "%s#%d" % (conn.types[best], conn.ids[best]), math.degrees(conn.pref_az[best]), best_n

        def centroid_az() -> float:
            """Spike-weighted preferred azimuth of the retina (circular mean)."""
            sx = sy = 0.0
            for i in retina:
                w = counts.get(i, 0)
                if w:
                    sx += w * math.cos(conn.pref_az[i])
                    sy += w * math.sin(conn.pref_az[i])
            return math.degrees(math.atan2(sy, sx)) if (sx or sy) else float("nan")

        def side_hz(want_side: str) -> float:
            idx = [i for i in visual if conn.sides[i] == want_side]
            return sum(counts.get(i, 0) for i in idx) / max(1, steps) / ctl.net.p.dt / max(1, len(idx))

        steering = sorted(set(DN_GROUPS["turn_left"]) | set(DN_GROUPS["turn_right"]))
        dn = [i for i, t in enumerate(conn.types) if t in steering]

        def dn_hz(want_side: str) -> float:
            idx = [i for i in dn if conn.sides[i] == want_side]
            return sum(counts.get(i, 0) for i in idx) / max(1, steps) / ctl.net.p.dt / max(1, len(idx))

        rname, rpref, rn = peak(retina)
        vname, vpref, vn = peak(visual)
        rows.append(dict(want_deg=want, aa_deg=me.geom_to_other.aa_deg,
                         retina_centroid_az=centroid_az(),
                         range_m=me.geom_to_other.range_m,
                         retina_peak=rname, retina_pref_az=rpref, retina_spikes=rn,
                         vis_peak=vname, vis_pref_az=vpref, vis_spikes=vn,
                         population_hz=ctl.net.population_rate(),
                         vis_left_hz=side_hz("L"), vis_right_hz=side_hz("R"),
                         dn_left_hz=dn_hz("L"), dn_right_hz=dn_hz("R")))
    return rows


def _verdicts(rows: List[dict]) -> List[str]:
    out = []
    pops = [r["population_hz"] for r in rows]
    out.append("population rate        %.4f - %.4f Hz   %s"
               % (min(pops), max(pops),
                  "OK" if max(pops) > 0.005 else "FAIL: silent -> raise bg_current / gain_photo"))
    az = [r["retina_pref_az"] for r in rows]
    cen = [r["retina_centroid_az"] for r in rows]
    out.append("retina peak cell       %s   %s"
               % (", ".join("%+.0f->%+.0f" % (r["aa_deg"], r["retina_pref_az"])
                            for r in rows),
                  "OK: tracks bearing" if len(set(az)) > 1 else "FAIL: bearing-blind"))
    out.append("retinal spike centre   %s   %s"
               % (", ".join("%+.0f->%+.1f" % (r["aa_deg"], r["retina_centroid_az"])
                            for r in rows),
                  "OK: follows the target" if len(rows) < 2 or
                  (cen[-1] - cen[0]) * (rows[-1]["aa_deg"] - rows[0]["aa_deg"]) > 0 else
                  "FAIL: does not move with the target"))
    d = [r["vis_right_hz"] - r["vis_left_hz"] for r in rows]
    mono = all(b >= a for a, b in zip(d, d[1:])) or all(b <= a for a, b in zip(d, d[1:]))
    out.append("visual L/R difference  %s   %s"
               % (" ".join("%+.4f" % v for v in d),
                  "OK: orders with bearing" if mono else
                  "FAIL: no bearing signal in the visual population"))
    dd = [r["dn_right_hz"] - r["dn_left_hz"] for r in rows]
    dmono = all(b >= a for a, b in zip(dd, dd[1:])) or all(b <= a for a, b in zip(dd, dd[1:]))
    span = max(dd) - min(dd)
    scale = max(abs(v) for v in dd) or 1.0
    out.append("descending L/R diff    %s   %s"
               % (" ".join("%+.4f" % v for v in dd),
                  # The lateral channel needs a *bearing-dependent* component, not
                  # just an offset that happens to be monotone: a fixed bias flies
                  # the aircraft into a fixed turn, it does not aim a gun.
                  "OK: orders with bearing (span %.1f of %.1f Hz)"
                  % (span, scale) if (dmono and span > 0.5 * scale) else
                  "FAIL: %s -> the target's bearing is mostly lost in the fan-in onto "
                  "the DNs (reported, not hidden; see docs/EXPERIMENTS.md 3)"
                  % ("flat" if span <= 1e-9 else "offset-dominated, bearing span only %.1f of %.1f Hz"
                     % (span, scale))))
    return out


def probe_brain(args) -> int:
    steps = getattr(args, "steps", 250)
    seed = getattr(args, "seed", 2)
    distance = getattr(args, "distance", 1200.0)
    side = getattr(args, "side", "blue")
    bearings = getattr(args, "bearings", None) or [60, 0, -60]
    use_torch = getattr(args, "use_torch", False)
    torch_device = getattr(args, "torch_device", None)
    plasticity = getattr(args, "plasticity", False)
    brain_kind = getattr(args, "brain", "synthetic")

    rows = _sweep(list(bearings), steps, seed, distance, side,
                  use_torch=use_torch, torch_device=torch_device,
                  plasticity=plasticity, brain_kind=brain_kind)
    print("connectome probe: %d bearings x %d steps (%.2f s of brain time each)"
          % (len(rows), steps, steps * 0.002))
    print("target %.0f m from the %s, both level, neither moving" % (distance, side))
    print()
    print("  cmd AA  measured AA   range   retinal centre   peak cell      pref  spikes   visual peak     pref  spikes  pop Hz   L Hz    R Hz   DN L-R")
    for r in rows:
        print("  %+5.0f   %+8.1f   %7.0f   %+11.1f   %-12s %+5.0f %7d   %-14s %+5s %6d  %7.4f %7.4f %7.4f  %+7.4f"
              % (r["want_deg"], r["aa_deg"], r["range_m"], r["retina_centroid_az"],
                 r["retina_peak"] or "-",
                 r["retina_pref_az"] if r["retina_pref_az"] is not None else float("nan"),
                 r["retina_spikes"], r["vis_peak"] or "-",
                 "%+.0f" % r["vis_pref_az"] if r["vis_pref_az"] is not None else "-",
                 r["vis_spikes"], r["population_hz"], r["vis_left_hz"], r["vis_right_hz"],
                 r["dn_right_hz"] - r["dn_left_hz"]))
    print()
    for v in _verdicts(rows):
        print(v)

    out = getattr(args, "out", None)
    if out:
        with open(out, "w") as fh:
            json.dump(dict(rows=rows, verdicts=_verdicts(rows)), fh, indent=2)
        print("\nwrote %s" % out)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="probe the connectome brain open-loop")
    ap.add_argument("--steps", type=int, default=250, help="brain ticks per bearing")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--distance", type=float, default=1200.0, help="target range, m")
    ap.add_argument("--side", default="blue", choices=["blue", "red"])
    ap.add_argument("--bearings", type=int, nargs="*", default=None,
                    help="commanded angles off the nose (right positive)")
    ap.add_argument("--out", default=None, help="write the table as JSON")
    ap.add_argument("--brain", default="synthetic", choices=["synthetic", "malecns"], help="brain kind")
    ap.add_argument("--use-torch", action="store_true", help="use TorchLIFNetwork")
    ap.add_argument("--torch-device", default=None, help="torch device")
    ap.add_argument("--plasticity", action="store_true", help="enable KC->MBON plasticity")
    return probe_brain(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
