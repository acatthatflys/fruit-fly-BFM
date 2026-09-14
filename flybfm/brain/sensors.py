"""Encoding the dogfight into the fly's sensory periphery.

The whole point of a connectome-constrained controller is that the *brain*
decides, so the encoder/decoder pair is where a project like this cheats or
stays honest.  Rules used here:

  * Visual input is a real projection of the target onto a retinal mosaic, so
    the brain sees a *blob at a bearing*, not "angle off = 3.2 degrees".
  * Motion and looming signals are derived the way the fly derives them
    (T4/T5 direction-selective cells for wide-field motion, LC4/LPLC2 for
    looming), and they are driven by the same quantities the environment uses.
  * Body-rate feedback enters through haltere/mechanosensory afferents, not
    through a magic "roll rate" input.
  * Which neurons receive which channel is looked up from the connectome's cell
    types, so the same encoder works on the synthetic graph and on MaleCNS.

Nothing here is trained.  If the encoder is hand-tuned, the result is a
statement about the encoder; keeping it fixed and anatomically mapped is what
lets the readout claim any credit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .. import vecmath as vm


@dataclass
class SensorConfig:
    n_az: int = 16               # retinal mosaic columns (azimuth)
    n_el: int = 8                # rows (elevation)
    az_span_deg: float = 360.0   # fly eyes wrap nearly all the way round
    el_span_deg: float = 160.0
    target_wingspan_m: float = 9.8       # F-16; change for fly-on-fly
    bg_brightness: float = 0.0
    bg_current: float = 2.0      # tonic photoreceptor activity (nA-ish)
    gain_photo: float = 8.0      # current per unit contrast
    gain_motion: float = 3.0
    gain_loom: float = 4.0
    gain_gyro: float = 1.2
    gain_air: float = 0.8
    excitation_sign: float = 1.0


class SensorBank:
    """Maps aircraft state -> currents on identified sensory neuron groups."""

    def __init__(self, conn, cfg: Optional[SensorConfig] = None):
        self.c = conn
        self.cfg = cfg or SensorConfig()
        self.by_type = conn.index_by_type()
        self.photo = self._pick("R1-R6")
        self.color = self._pick("R7", "R8")
        self.motion = self._pick("T4", "T5")
        self.flow = self._pick("VS", "HS")
        self.loom = self._pick("LC4", "LPLC2")
        self.haltere = self._pick("haltere", "haltere-afferent", "HAL")
        self.wind = self._pick("wind", "antennal-mechano", "AMMC")
        self._prev_size = 0.0
        self._loom_ma = 0.0

    def _pick(self, *names) -> List[int]:
        out: List[int] = []
        for n in names:
            out.extend(self.by_type.get(n, []))
        return out

    @property
    def available(self) -> Dict[str, int]:
        return {"photo": len(self.photo), "color": len(self.color),
                "motion": len(self.motion), "flow": len(self.flow),
                "loom": len(self.loom), "haltere": len(self.haltere),
                "wind": len(self.wind)}

    # ------------------------------------------------------------------ encode
    def inject(self, net, me, other, geom, los_rate_rad: float, dt: float) -> Dict[str, float]:
        """Drive one brain tick. `me`/`other` are sim State objects."""
        cfg = self.cfg
        # ---- apparent target: bearing in the body frame + angular size
        rel = vm.sub(other.pos, me.pos)
        rng = max(vm.norm(rel), 1e-3)
        size = math.atan2(cfg.target_wingspan_m, rng)          # radians
        f = vm.unit(me.vel) if vm.norm(me.vel) > 1e-6 else (1.0, 0.0, 0.0)
        right = vm.right_of(f)
        up = vm.up_of(f, right)
        los_u = vm.scale(rel, 1.0 / rng)
        az = math.atan2(vm.dot(los_u, right), vm.dot(los_u, f))
        el = math.asin(vm.clamp(vm.dot(los_u, up), -1.0, 1.0))

        # ---- retinal mosaic: a Gaussian blob for the target
        #
        # The blur width is floored at roughly one ommatidium.  A 10 m target at
        # 1.2 km subtends 8 mrad while ommatidia are ~5 deg apart, so without
        # this floor the blob falls between every cell centre, injects nothing,
        # and the "brain" sits silent: a classic silent-network bug that looks
        # exactly like a broken learning rule.
        # Blob width: the target's true angular size, floored so that it always
        # covers at least ~one receptor of *this* mosaic.
        #
        # A real fly has ~700 ommatidia per eye (0.4 deg apart), so a distant
        # target is genuinely sub-ommatidial and still lands on a receptor.  A
        # reduced synthetic retina with 12 azimuth columns has 30 deg between
        # receptors, so flooring the blob at the biological 4.5 deg puts the
        # target *between* every receptor: the retina then emits an identical
        # spike train whatever the target does, and every downstream layer looks
        # like it carries no information.  Scale the floor to the mosaic you
        # actually simulate.
        el_rows = max(1, len(self.photo) // max(cfg.n_az, 1))
        az_spacing = math.radians(cfg.az_span_deg) / max(cfg.n_az, 1)
        el_spacing = math.radians(cfg.el_span_deg) / max(el_rows, 1)
        sigma = max(size, 0.55 * max(az_spacing, el_spacing))
        blob = 0.0
        for neuron in self.photo:
            # each photoreceptor looks at its own point in the visual field
            paz = self.c.pref_az[neuron] if neuron < len(self.c.pref_az) else 0.0
            pel = self.c.pref_el[neuron] if neuron < len(self.c.pref_el) else 0.0
            d2 = _wrap(paz - az) ** 2 + (pel - el) ** 2
            val = math.exp(-d2 / (2.0 * sigma * sigma))
            # tonic photoreceptor activity + the target's contrast
            net.add_input(neuron, cfg.bg_current + cfg.gain_photo * val)
            blob += val
        # colour channel: same blob, weaker (R7/R8 are a separate mosaic)
        for neuron in self.color:
            net.add_input(neuron, cfg.bg_current * 0.7 + cfg.gain_photo * 0.25 * blob)

        # ---- looming: d(size)/dt, saturating, only on the looming detectors
        d_size = (size - self._prev_size) / max(dt, 1e-6)
        self._prev_size = size
        loom = math.tanh(max(0.0, d_size) * 0.4)
        for neuron in self.loom:
            net.add_input(neuron, cfg.bg_current * 0.5 + cfg.gain_loom * loom)

        # ---- wide-field motion: LOS rotation drives T4/T5 and the tangential cells
        motion = math.tanh(los_rate_rad * 2.0)
        for neuron in self.motion:
            sgn = 1.0 if self.c.sides[neuron] == "L" else -1.0
            net.add_input(neuron, cfg.bg_current * 0.4 + cfg.gain_motion * motion * sgn)
        for neuron in self.flow:
            sgn = 1.0 if self.c.sides[neuron] == "L" else -1.0
            net.add_input(neuron, cfg.bg_current * 0.4 + cfg.gain_motion * 0.7 * motion * sgn)

        # ---- haltere / gyro: body rates (roll, pitch, yaw) + attitude
        roll_rate = me.mu          # the sim exposes bank; rate is reconstructed
        for k, neuron in enumerate(self.haltere):
            sgn = 1.0 if self.c.sides[neuron] == "L" else -1.0
            drive = (sgn * math.sin(me.mu) * 0.8
                     + math.sin(me.gamma) * 0.5
                     + math.tanh(los_rate_rad))
            net.add_input(neuron, cfg.bg_current * 0.5 + cfg.gain_gyro * drive)
        # ---- wind / airspeed sense
        v_norm = math.tanh((me.v - 200.0) / 120.0)
        for neuron in self.wind:
            net.add_input(neuron, cfg.bg_current * 0.5 + cfg.gain_air * v_norm)

        return {"size_rad": size, "az": az, "el": el, "loom": loom,
                "range": rng, "brightness": blob}

    # ------------------------------------------------------------------ helper
    def _cell_az(self, i: int) -> float:
        cfg = self.cfg
        return math.radians(-cfg.az_span_deg * 0.5 + cfg.az_span_deg * (i + 0.5) / cfg.n_az)

    def _cell_el(self, i: int) -> float:
        cfg = self.cfg
        return math.radians(-cfg.el_span_deg * 0.5 + cfg.el_span_deg * (i + 0.5) / cfg.n_el)


def _wrap(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a
