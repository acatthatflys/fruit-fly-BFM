"""Connectome-constrained brain tests.

The important one is `test_retina_tracks_bearing`: it is the cheapest test that
catches the whole class of "the brain has no bearing sense" bugs, all four of
which have the same symptom and none of which look like a bug in a trace.
"""
from __future__ import annotations

import math
import unittest

from flybfm.brain.controller import ReadoutParams, build_controller
from flybfm.brain.lif import LIFNetwork, LIFParams
from flybfm.brain.sensors import SensorBank, SensorConfig
from flybfm.sim.arena import Scenario
from flybfm.sim.aircraft import Command
from flybfm.sim.dogfight import Dogfight, EnvConfig
from flybfm.sim.scripted import POOL_BY_NAME


class TestConnectome(unittest.TestCase):
    def test_subgraph_is_sparse_and_labelled(self):
        ctl = build_controller("synthetic", seed=2)
        c = ctl.conn
        self.assertGreater(len(c), 300)
        self.assertGreater(c.n_edges, len(c))
        stats = c.stats()
        self.assertLess(stats["mean_out_degree"], 20.0)
        types = set(c.types)
        for expected in ("R1-R6", "L1-L5", "T4", "T5", "VPN", "DNg02", "STMD"):
            self.assertIn(expected, types, f"missing cell type {expected}")

    def test_neurons_have_preferred_azimuths(self):
        ctl = build_controller("synthetic", seed=1)
        az = [a for a, t in zip(ctl.conn.pref_az, ctl.conn.types) if t.startswith("R1-")]
        self.assertTrue(az)
        self.assertLess(min(az), -1.0)             # eye covers left and right
        self.assertGreater(max(az), 1.0)

    def test_lif_network_spikes_with_tonic_drive(self):
        ctl = build_controller("synthetic", seed=3)
        net = ctl.net
        total = 0
        for _ in range(200):
            net.clear_inputs()
            for i in range(len(ctl.conn)):
                net.set_input(i, 1.5)
            total += len(net.step())
        self.assertGreater(total, 0, "network is silent under drive")
        self.assertGreater(net.population_rate(), 0.0)

    def test_plasticity_path_exists(self):
        """mark_plastic must find the KC -> MBON edges if they are present."""
        ctl = build_controller("synthetic", seed=2)
        n = ctl.net.mark_plastic(("KC",), ("MBON",))
        self.assertGreater(n, 0, "no KC -> MBON edges in the mushroom body subgraph")
        # three-factor rule: no coincidence (eligibility 0) -> no weight change,
        # even at full dopamine.  This is the property that makes the rule a
        # *learning* rule rather than a global gain.
        edges = ctl.net.plastic_edges[:5]
        before = [ctl.conn.weight[k] for k in edges]
        ctl.net.deliver_dopamine(1.0, lr=0.5, weight_decay=0.0)
        self.assertEqual(before, [ctl.conn.weight[k] for k in edges])
        # with an eligibility trace present, the same dopamine changes the weight
        for k in edges:
            ctl.net.elig[k] = 1.0
        ctl.net.deliver_dopamine(1.0, lr=0.5, weight_decay=0.0)
        after = [ctl.conn.weight[k] for k in edges]
        self.assertNotEqual(before, after)
        # and punishment moves it the other way
        for k in edges:
            ctl.net.elig[k] = 1.0
        mid = [ctl.conn.weight[k] for k in edges]
        ctl.net.deliver_dopamine(-1.0, lr=0.5, weight_decay=0.0)
        for k, m in zip(edges, mid):
            self.assertLess(ctl.conn.weight[k], m)


class TestSensors(unittest.TestCase):
    def _peak_bearing(self, ctl, env, bearing_deg, steps=200):
        """Return (measured nose error, peak-response receptor azimuth, spikes)."""
        env.reset(Scenario(range_m=1200.0, taa_deg=0.0, nose_offset_deg=bearing_deg,
                           alt_m=6000.0, v_a=280.0, v_t=250.0, tag="test"))
        blue, red = env.blue, env.red
        net = ctl.net
        net.rate_ma = [0.0] * net.n
        counts = [0] * net.n
        for k in range(steps):
            if k % 5 == 0:
                net.clear_inputs()
                ctl.sensors.inject(net, blue.s, red.s, blue.geom_to_other, 0.0, net.p.dt)
            for i in net.step():
                counts[i] += 1
        idx = [i for i, t in enumerate(ctl.conn.types) if t.startswith("R1-")]
        best = max(idx, key=lambda i: counts[i])
        return (blue.geom_to_other.aa_deg, math.degrees(ctl.conn.pref_az[best]),
                counts[best], sum(counts[i] for i in idx))

    def test_retina_tracks_bearing(self):
        """The peak-response photoreceptor must follow the target's bearing.

        Failure modes that this catches, all observed in this repo:
          * the target's image is smaller than the receptor spacing, so the
            retina emits the *same* spike train for every bearing;
          * the sensor's cell coordinates are not the receptive fields';
          * the drive is below threshold, so nothing spikes at all.
        """
        ctl = build_controller("synthetic", seed=2)
        env = Dogfight(EnvConfig(decision_dt=0.1, max_time_s=2.0))
        samples = []
        for b in (-60.0, 0.0, 60.0):
            aa, peak, best_count, total = self._peak_bearing(ctl, env, b)
            self.assertGreater(total, 0, f"no photoreceptor spikes at bearing {b}")
            samples.append((aa, peak))
        samples.sort()                                   # by measured nose error
        self.assertLess(samples[0][0], samples[1][0])
        # the peak-response receptor must move the same way as the target
        self.assertLess(samples[0][1], samples[1][1],
                        f"peak response did not follow the target: {samples}")
        self.assertLess(samples[1][1], samples[2][1],
                        f"peak response did not follow the target: {samples}")
        self.assertGreater(samples[2][1] - samples[0][1], 20.0,
                           f"peak moved only {samples}")


class TestController(unittest.TestCase):
    def test_readout_round_trips(self):
        p = ReadoutParams(turn_gain=25.0, vis_gain=6.0, roll_sign=-1.0)
        q = ReadoutParams.from_vector(p.vector())
        self.assertEqual(p.vector(), q.vector())
        self.assertEqual(ReadoutParams.N_PARAMS, len(p.vector()))
        m = p.mutate(__import__("random").Random(0), sigma=0.1)
        self.assertEqual(len(m.vector()), ReadoutParams.N_PARAMS)

    def test_controller_flies_a_fight(self):
        ctl = build_controller("synthetic", seed=2)
        env = Dogfight(EnvConfig(decision_dt=0.2, max_time_s=2.0))
        env.reset(Scenario(range_m=1200.0, taa_deg=30.0, alt_m=6000.0))
        opp = POOL_BY_NAME["level"](seed=3)
        n = 0
        while not env.done:
            cb = ctl.act(env.blue, env)
            ctl.observe(env.observation_vector(env.blue), 0.0)
            self.assertIsInstance(cb, Command)
            self.assertTrue(-1.0 <= cb.roll <= 1.0)
            self.assertTrue(-1.0 <= cb.pull <= 1.0)
            self.assertTrue(0.0 <= cb.throttle <= 1.0)
            env.step(cb, opp(env.red, env))
            n += 1
            if n > 20:
                break
        st = ctl.stats
        self.assertGreater(st["brain_steps"], 0)
        self.assertGreater(len(env.frames), 1)

    def test_dn_groups_split_by_side(self):
        """Summing left and right copies of the same descending neuron deletes the
        lateral signal -- the readout must keep them apart."""
        ctl = build_controller("synthetic", seed=2)
        g = ctl._groups()
        self.assertTrue(g["turn_left"] and g["turn_right"])
        self.assertFalse(set(g["turn_left"]) & set(g["turn_right"]))
        self.assertTrue(g["vis_left"] and g["vis_right"])
        self.assertFalse(set(g["vis_left"]) & set(g["vis_right"]))

    def test_describe_is_serialisable(self):
        import json
        ctl = build_controller("synthetic", seed=2)
        json.dumps(ctl.describe())


if __name__ == "__main__":
    unittest.main()
