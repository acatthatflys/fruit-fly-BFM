"""Simulator and ballistics tests. Run: python -m unittest discover -s tests -v"""
from __future__ import annotations

import math
import unittest

from flybfm.sim.aircraft import G, Aircraft, Command, State, default_f16
from flybfm.sim.arena import CANONICAL_SETUPS, Scenario, ScenarioSpace
from flybfm.sim.dogfight import Dogfight, EnvConfig, run_match
from flybfm.sim.geometry import compute
from flybfm.sim.gun import (Gun, GunState, HIT_RADIUS_M, lead_solution,
                            required_lead_angle_deg)
from flybfm.sim.scripted import POOL_BY_NAME


class TestAircraft(unittest.TestCase):
    def test_corner_and_sustained_speed(self):
        p = default_f16()
        # q = n W / (S Cl_max); V = sqrt(2q/rho)
        expect = math.sqrt(2.0 * p.n_max * p.mass_kg * G
                           / (p.wing_area_m2 * p.cl_max * 1.225))
        self.assertAlmostEqual(p.corner_speed(), expect, places=3)
        self.assertTrue(150.0 < p.corner_speed() < 200.0)
        self.assertLess(p.sustained_turn_speed(), p.corner_speed())

    def test_level_turn_rate_matches_analytic(self):
        """The textbook level-turn rate g*sqrt(n^2-1)/V applies when the bank angle
        is exactly the one that holds altitude, cos(mu) = 1/n -- not at mu = 90 deg,
        where the aircraft is falling."""
        p = default_f16()
        ac = Aircraft(p, State(pos=(0.0, 0.0, 6000.0), v=250.0, psi=0.0, gamma=0.0, mu=0.0))
        ac.s.n = 9.0
        ac.s.mu = math.acos(1.0 / 9.0)
        expect = math.degrees(G * math.sqrt(9.0 ** 2 - 1.0) / 250.0)
        self.assertAlmostEqual(abs(ac.turn_rate_deg_s()), expect, delta=0.1)
        # and the model's own law is psi_dot = g n sin(mu) / (V cos(gamma))
        self.assertAlmostEqual(abs(ac.turn_rate_deg_s()),
                               math.degrees(G * 9.0 * math.sin(ac.s.mu) / 250.0), places=6)

    def test_ninety_degree_turn_timing(self):
        """Banked for a level 9 g turn at 250 m/s: 90 deg of heading in ~4.5 s of
        *steady* turn, plus ~1.5 s to wind the load factor on at 9 g/s."""
        p = default_f16()
        ac = Aircraft(p, State(pos=(0.0, 0.0, 8000.0), v=250.0, psi=0.0, gamma=0.0,
                               mu=math.acos(1.0 / 9.0)))
        dt = 0.02
        t = 0.0
        while t < 60.0:
            ac.step(Command(roll=0.0, pull=1.0, throttle=1.0, trigger=False), dt)
            t += dt
            if abs(ac.s.psi) >= math.pi / 2:
                break
        self.assertTrue(4.0 < t < 7.0, f"90 deg of turn took {t:.2f}s")

    def test_stall_and_ground(self):
        p = default_f16()
        ac = Aircraft(p, State(pos=(0.0, 0.0, 300.0), v=60.0, gamma=math.radians(-60.0)))
        for _ in range(200):
            ac.step(Command(roll=0.0, pull=-1.0, throttle=0.0, trigger=False), 0.02)
        self.assertFalse(ac.s.alive)


def vm_angle(psi: float) -> float:
    return abs(psi)


class TestGeometry(unittest.TestCase):
    def test_tail_chase_aspect(self):
        """Aspect is measured at the target: 0 = shooter on the target's tail."""
        a = State(pos=(0.0, 0.0, 6000.0), v=250.0, psi=0.0)
        b = State(pos=(1000.0, 0.0, 6000.0), v=250.0, psi=0.0)
        g = compute(a, b)
        self.assertLess(g.taa_deg, 1.0)            # on his tail
        self.assertLess(abs(g.aa_deg), 1.0)
        self.assertLess(g.hca_deg, 1.0)
        self.assertEqual(g.aspect_sign, 1)
        self.assertTrue(g.in_rear_hemisphere)

    def test_beam_aspect(self):
        a = State(pos=(0.0, 0.0, 6000.0), v=250.0, psi=0.0)
        b = State(pos=(1000.0, 0.0, 6000.0), v=250.0, psi=-math.pi / 2)
        self.assertAlmostEqual(compute(a, b).taa_deg, 90.0, places=6)

    def test_head_on(self):
        a = State(pos=(0.0, 0.0, 6000.0), v=250.0, psi=0.0)
        b = State(pos=(1000.0, 0.0, 6000.0), v=250.0, psi=math.pi)
        g = compute(a, b)
        self.assertAlmostEqual(g.hca_deg, 180.0, places=3)
        self.assertAlmostEqual(g.taa_deg, 180.0, places=3)
        self.assertGreater(g.closure_ms, 0.0)

    def test_closure_sign(self):
        """closure is + when the range is shrinking, for *both* aircraft."""
        a = State(pos=(0.0, 0.0, 6000.0), v=300.0, psi=0.0)      # faster, behind
        b = State(pos=(1000.0, 0.0, 6000.0), v=200.0, psi=0.0)
        self.assertGreater(compute(a, b).closure_ms, 0.0)
        self.assertGreater(compute(b, a).closure_ms, 0.0)
        self.assertAlmostEqual(compute(a, b).closure_ms, compute(b, a).closure_ms, places=6)
        # and a target running away faster than we close gives negative closure
        slow = State(pos=(0.0, 0.0, 6000.0), v=200.0, psi=0.0)
        fast_ahead = State(pos=(1000.0, 0.0, 6000.0), v=300.0, psi=0.0)
        self.assertLess(compute(slow, fast_ahead).closure_ms, 0.0)

    def test_range_is_symmetric(self):
        a = State(pos=(0.0, 0.0, 6000.0), v=250.0, psi=0.3)
        b = State(pos=(800.0, -400.0, 6400.0), v=220.0, psi=-1.0)
        self.assertAlmostEqual(compute(a, b).range_m, compute(b, a).range_m, places=6)


class TestGun(unittest.TestCase):
    def _perfect_shot_hit_rate(self, range_m, v_own, v_tgt, n=500, seed=3):
        """Slave the nose exactly to the ballistic lead point and see what lands."""
        gun = Gun(GunState(ammo=10 ** 6), rounds_per_s=1.0 / 0.06, seed=seed)
        hits = 0
        for k in range(n):
            sp = (0.0, 0.0, 6000.0)
            sv = (v_own, 0.0, 0.0)
            tp = (range_m, 0.0, 6000.0)
            tv = (v_tgt, 0.0, 0.0)
            lead, _ = lead_solution(sp, sv, tp, tv)
            d = (lead[0] - sp[0], lead[1] - sp[1], lead[2] - sp[2])
            nrm = math.sqrt(sum(x * x for x in d))
            sv_aim = tuple(x / nrm * v_own for x in d)      # nose on the lead point
            _, h = gun.fire(sp, sv_aim, tp, tv, 0.06, True)
            hits += h
        return hits / n

    def test_ballistics_stern_chase(self):
        # measured with perfect aim: 99.8% at 500 m, 93.8% at 700 m, 80% at 900 m
        self.assertGreater(self._perfect_shot_hit_rate(500.0, 300.0, 250.0), 0.90)
        self.assertGreater(self._perfect_shot_hit_rate(900.0, 300.0, 250.0), 0.70)
        # and it must degrade with range, not stay flat
        far = self._perfect_shot_hit_rate(3000.0, 300.0, 250.0, n=200)
        near = self._perfect_shot_hit_rate(500.0, 300.0, 250.0, n=200)
        self.assertLess(far, near)

    def test_lead_point_uses_relative_velocity_and_aims_high(self):
        """The aim direction is toward `target + (v_t - v_s) t`, and it is *above*
        the target's future position because the round drops on the way."""
        sp, sv = (0.0, 0.0, 6000.0), (300.0, 0.0, 0.0)
        tp, tv = (900.0, 0.0, 6000.0), (250.0, 0.0, 0.0)
        lead, t_f = lead_solution(sp, sv, tp, tv)
        expect = (900.0 + (250.0 - 300.0) * t_f)           # relative velocity term
        self.assertAlmostEqual(lead[0], expect, delta=2.0)
        self.assertGreater(lead[2], 6000.0)                # aim high: the round falls
        self.assertAlmostEqual(lead[2] - 6000.0, 0.5 * 9.80665 * t_f ** 2, delta=1.0)

    def test_lead_point_with_a_stationary_shooter(self):
        """With the shooter at rest the lead point is simply the future position."""
        lead, t_f = lead_solution((0.0, 0.0, 6000.0), (0.0, 0.0, 0.0),
                                  (900.0, 0.0, 6000.0), (250.0, 0.0, 0.0))
        self.assertAlmostEqual(lead[0], 900.0 + 250.0 * t_f, delta=2.0)

    def test_required_lead_is_small_in_a_tail_chase(self):
        # a co-linear stern chase needs only a few degrees of lead
        la = required_lead_angle_deg((0.0, 0.0, 6000.0), (300.0, 0.0, 0.0),
                                     (700.0, 0.0, 6000.0), (250.0, 0.0, 0.0))
        self.assertLess(abs(la), 12.0)

    def test_dispersion_scales_with_range(self):
        """A fixed angular dispersion must produce a range-dependent hit rate."""
        short = self._perfect_shot_hit_rate(300.0, 300.0, 250.0, n=300)
        long_ = self._perfect_shot_hit_rate(1500.0, 300.0, 250.0, n=300)
        self.assertGreater(short, long_)


class TestScenarioAndEnv(unittest.TestCase):
    def test_scenario_space_randomises_everything(self):
        import random
        sp = ScenarioSpace()
        rng = random.Random(0)
        seen = set()
        for _ in range(30):
            s = sp.sample(rng)
            seen.add((round(s.range_m / 100), round(s.taa_deg / 5), round(s.alt_m / 100)))
        self.assertGreater(len(seen), 25)          # not collapsing to a few setups

    def test_canonical_setups_all_run_and_terminate(self):
        cfg = EnvConfig(max_time_s=12.0)
        for tag, sc in CANONICAL_SETUPS.items():
            env = Dogfight(cfg)
            env.reset(sc)
            a = POOL_BY_NAME["guns"](seed=1)
            b = POOL_BY_NAME["level"](seed=2)
            while not env.done:
                env.step(a(env.blue, env), b(env.red, env))
            self.assertIsNotNone(env.result, tag)
            # the loop can overrun by at most one decision interval
            self.assertLess(env.t, 12.0 + cfg.decision_dt + 1e-9, tag)

    def test_frames_carry_what_the_viewer_needs(self):
        s, tr, env = run_match(CANONICAL_SETUPS["perch"],
                               POOL_BY_NAME["guns"](seed=1), POOL_BY_NAME["level"](seed=2),
                               EnvConfig(max_time_s=6.0))
        self.assertTrue(env.frames)
        f = env.frames[-1]
        for key in ("t", "b", "r", "g", "hp"):
            self.assertIn(key, f)
        for key in ("p", "psi", "gam", "v", "mu", "n", "thr", "ammo", "cmd", "es"):
            self.assertIn(key, f["b"])
        self.assertIsInstance(f["b"]["cmd"], list)
        self.assertEqual(len(f["b"]["cmd"]), 4)

    def test_replay_round_trips(self):
        s, tr, env = run_match(CANONICAL_SETUPS["perch"],
                               POOL_BY_NAME["guns"](seed=1), POOL_BY_NAME["level"](seed=2),
                               EnvConfig(max_time_s=6.0))
        rp = env.to_replay()
        import json
        blob = json.dumps(rp)
        self.assertGreater(len(blob), 1000)
        self.assertIn("frames", json.loads(blob))


if __name__ == "__main__":
    unittest.main()
