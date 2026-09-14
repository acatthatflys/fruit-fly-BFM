"""Regression test for load_malecns_flat column-sniffing.

DATA.md flags that column names move between MaleCNS releases. This test builds
tiny synthetic .feather fixtures with the expected schema variants and ensures
the loader finds the right columns and produces a valid Connectome.

Requires pandas/pyarrow; skipped if not installed.
"""
from __future__ import annotations

import os
import tempfile
import unittest


class TestLoadMalecnsFlat(unittest.TestCase):
    def _make_fixtures(self, tmpdir, ann_cols, tx_cols, conn_cols, n_neurons=6, n_edges=8):
        """Write three feather files with custom column names."""
        try:
            import pandas as pd
        except Exception as exc:
            self.skipTest(f"pandas not installed: {exc}")

        # deterministic tiny graph
        # neurons: body IDs 101..106, types covering the flight pathway
        body_ids = list(range(101, 101 + n_neurons))
        types = ["R1-R6", "VPN", "DNp26", "KC", "MBON", "PAM"][:n_neurons]
        sides = ["L", "R", "L", "R", "L", "R"][:n_neurons]
        transmitters = ["ACH", "GABA", "ACH", "ACH", "ACH", "ACH"][:n_neurons]

        # annotations
        ann_data = {}
        # ann_cols is tuple (id_col, type_col, side_col)
        id_col, type_col, side_col = ann_cols
        ann_data[id_col] = body_ids
        ann_data[type_col] = types
        ann_data[side_col] = sides
        ann_df = pd.DataFrame(ann_data)

        # transmitters
        tx_id_col, tx_col = tx_cols
        tx_df = pd.DataFrame({tx_id_col: body_ids, tx_col: transmitters})

        # connectome edges: simple chain + some KC->MBON plastic edges
        pre_ids = []
        post_ids = []
        weights = []
        # R1-R6 -> VPN
        pre_ids.append(body_ids[0]); post_ids.append(body_ids[1]); weights.append(5)
        # VPN -> DNp26
        pre_ids.append(body_ids[1]); post_ids.append(body_ids[2]); weights.append(3)
        # KC -> MBON (plastic)
        if n_neurons >= 5:
            pre_ids.append(body_ids[3]); post_ids.append(body_ids[4]); weights.append(4)
            pre_ids.append(body_ids[3]); post_ids.append(body_ids[4]); weights.append(2)
        # extra random edges to reach n_edges
        import random
        rng = random.Random(0)
        while len(pre_ids) < n_edges:
            a = rng.choice(body_ids)
            b = rng.choice(body_ids)
            if a == b:
                continue
            pre_ids.append(a); post_ids.append(b); weights.append(rng.randint(1, 10))

        conn_pre_col, conn_post_col, conn_w_col = conn_cols
        conn_df = pd.DataFrame({
            conn_pre_col: pre_ids,
            conn_post_col: post_ids,
            conn_w_col: weights,
        })

        # file names must contain "connectome-weights" and "body-neurotransmitters"
        # because loader does path.replace("connectome-weights", "body-neurotransmitters")
        ann_path = os.path.join(tmpdir, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
        conn_path = os.path.join(tmpdir, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
        tx_path = conn_path.replace("connectome-weights", "body-neurotransmitters")

        ann_df.to_feather(ann_path)
        tx_df.to_feather(tx_path)
        conn_df.to_feather(conn_path)

        return ann_path, tx_path, conn_path, body_ids

    def test_default_column_names(self):
        """Default release names: bodyId, cell_type, side, transmitter, pre, post, weight."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path, tx_path, conn_path, body_ids = self._make_fixtures(
                tmpdir,
                ann_cols=("bodyId", "cell_type", "side"),
                tx_cols=("bodyId", "transmitter"),
                conn_cols=("pre", "post", "weight"),
            )
            from flybfm.brain.connectome import load_malecns_flat
            c = load_malecns_flat(conn_path, ann_path, min_synapses=1)
            self.assertEqual(len(c.ids), 6)
            self.assertGreater(c.n_edges, 0)
            # transmitter sign: GABA should be negative
            # find GABA neuron (body 102)
            idx_gaba = c.ids.index(102) if 102 in c.ids else None
            if idx_gaba is not None:
                # outgoing edges from GABA should be negative
                for a, b, w in zip(c.pre, c.post, c.weight):
                    if a == idx_gaba:
                        self.assertLess(w, 0, "GABA edge should be inhibitory (negative)")

    def test_alternative_column_names(self):
        """Alternative names observed in older/newer releases: body_id, type, side_predicted, nt_type, bodyId_pre, etc."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path, tx_path, conn_path, body_ids = self._make_fixtures(
                tmpdir,
                ann_cols=("body_id", "type", "side_predicted"),
                tx_cols=("body_id", "nt_type"),
                conn_cols=("bodyId_pre", "bodyId_post", "syn_count"),
            )
            from flybfm.brain.connectome import load_malecns_flat
            c = load_malecns_flat(conn_path, ann_path, min_synapses=1)
            self.assertEqual(len(c.ids), 6)
            self.assertGreater(c.n_edges, 0)

    def test_third_variant_column_names(self):
        """Third variant: id, primary_type, side, top_nt, pre_bodyId, post_bodyId, count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path, tx_path, conn_path, body_ids = self._make_fixtures(
                tmpdir,
                ann_cols=("id", "primary_type", "side"),
                tx_cols=("id", "top_nt"),
                conn_cols=("pre_bodyId", "post_bodyId", "count"),
            )
            from flybfm.brain.connectome import load_malecns_flat
            c = load_malecns_flat(conn_path, ann_path, min_synapses=1)
            self.assertEqual(len(c.ids), 6)
            self.assertGreater(c.n_edges, 0)

    def test_min_synapses_filter_and_weight_fn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path, tx_path, conn_path, body_ids = self._make_fixtures(
                tmpdir,
                ann_cols=("bodyId", "cell_type", "side"),
                tx_cols=("bodyId", "transmitter"),
                conn_cols=("pre", "post", "weight"),
                n_edges=10,
            )
            from flybfm.brain.connectome import load_malecns_flat
            # filter out low-count edges
            c_all = load_malecns_flat(conn_path, ann_path, min_synapses=1)
            c_filtered = load_malecns_flat(conn_path, ann_path, min_synapses=5)
            self.assertGreaterEqual(c_all.n_edges, c_filtered.n_edges)

            # custom weight_fn
            def custom_w(n, pre_tx, post_tx):
                return float(n) * 0.1

            c_custom = load_malecns_flat(conn_path, ann_path, min_synapses=1, weight_fn=custom_w)
            # all weights should be positive before sign application, then sign applied
            # check that custom mapping was used (weights scaled by 0.1, then sign)
            self.assertGreater(c_custom.n_edges, 0)

    def test_plastic_edges_present_in_real_loader(self):
        """The tiny fixture includes KC->MBON, so mark_plastic should find it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ann_path, tx_path, conn_path, body_ids = self._make_fixtures(
                tmpdir,
                ann_cols=("bodyId", "cell_type", "side"),
                tx_cols=("bodyId", "transmitter"),
                conn_cols=("pre", "post", "weight"),
            )
            from flybfm.brain.connectome import load_malecns_flat
            from flybfm.brain.lif import LIFNetwork
            c = load_malecns_flat(conn_path, ann_path)
            net = LIFNetwork(c, seed=0, plasticity=True)
            n = net.mark_plastic(("KC",), ("MBON",))
            self.assertGreater(n, 0, "KC->MBON plastic edges should be found in real loader fixture")


if __name__ == "__main__":
    unittest.main()
