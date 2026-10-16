import os
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import pandas as pd

from swine_inference import distance, load_observations, simulate


class InferenceTest(unittest.TestCase):
    def test_simulation_works_outside_checkout_and_preserves_caller_randomness(self):
        parameters = {"trans": 2.0, "trans_betwn_herds": 1.0, "trans_neighbor": 1.0}
        np.random.seed(42)
        random.seed(42)
        numpy_state, python_state = np.random.get_state(), random.getstate()
        expected_numpy = np.random.random()
        expected_python = random.random()
        expected_next_numpy = np.random.random()
        expected_next_python = random.random()
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        original_directory = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                first = simulate(parameters, steps=4, seed=2026)
                self.assertEqual(np.random.random(), expected_numpy)
                self.assertEqual(random.random(), expected_python)
                second = simulate(parameters, steps=4, seed=2026)
                self.assertEqual(np.random.random(), expected_next_numpy)
                self.assertEqual(random.random(), expected_next_python)
                self.assertEqual(list(Path(directory).iterdir()), [])
        finally:
            os.chdir(original_directory)
        # EMULSION hashes agents by object identity; seeded streams therefore
        # do not imply identical trajectories. Both runs must remain valid,
        # and the exact caller random-stream checks above must hold.
        for trajectory in (first, second):
            self.assertEqual(trajectory.shape, (5,))
            self.assertTrue(np.isfinite(trajectory).all())
            self.assertTrue(((0 <= trajectory) & (trajectory <= 12)).all())

    def test_swiavs_adapter_accepts_all_four_paper_parameters(self):
        trajectory = simulate({
            "trans": 17.01, "trans_betwn_herds": 9.87,
            "trans_neighbor": 9.87, "trans_aux": 0.1,
        }, steps=2, seed=2026, model_variant="swiavs")
        self.assertEqual(trajectory.shape, (3,))
        self.assertTrue(np.isfinite(trajectory).all())
        self.assertTrue(((0 <= trajectory) & (trajectory <= 12)).all())

    def test_observations_preserve_order_and_reject_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.csv"
            pd.DataFrame({"week": [2, 0, 1], "nb_I": [5, 0, 3]}).to_csv(path, index=False)
            np.testing.assert_array_equal(load_observations(path, steps=2), [0, 3, 5])
            for data in [
                {"week": [0, 1], "nb_I": [0, 1]},
                {"week": [0, 1, 1], "nb_I": [0, 1, 2]},
                {"week": [0, 1, 2], "nb_I": [0, 1.5, 2]},
                {"week": [0, 1, 2], "nb_I": [0, 13, 2]},
                {"week": [0, 1, 2], "nb_I": [0, np.nan, 2]},
                {"week": [0, 1, 2], "wrong_column": [0, 1, 2]},
            ]:
                with self.subTest(data=data):
                    pd.DataFrame(data).to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        load_observations(path, steps=2)

    def test_distance_rejects_mismatched_shapes_and_nonfinite_values(self):
        self.assertEqual(distance({"infected_pens": [0, 3]}, {"infected_pens": [1, 1]}), 3)
        for values in [[1], [1, float("nan")], [[1, 2]]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                distance({"infected_pens": values}, {"infected_pens": [0, 1]})

    def test_invalid_parameters_fail_before_simulation(self):
        with self.assertRaises(ValueError):
            simulate({"trans": 1}, steps=2)
        with self.assertRaises(ValueError):
            simulate({"trans": -1, "trans_betwn_herds": 1, "trans_neighbor": 1}, steps=2)
