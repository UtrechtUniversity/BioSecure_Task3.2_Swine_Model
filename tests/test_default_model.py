"""Run with: python -m unittest discover -s tests -v."""

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class DefaultModelRegressionTest(unittest.TestCase):
    def test_default_model_completes_and_keeps_animals_in_assigned_herds(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            environment = os.environ.copy()
            environment.update(
                MPLBACKEND="Agg",
                MPLCONFIGDIR=str(temporary / "matplotlib"),
                XDG_CACHE_HOME=str(temporary / "cache"),
                PYTHONDONTWRITEBYTECODE="1",
            )
            result = subprocess.run(
                [
                    sys.executable, "-m", "emulsion", "run",
                    "swine_model_weekly-12fatpens-FINAL.yaml",
                    "--runs", "2", "--time", "3", "--seed", "123", "--quiet",
                    "--output-dir", str(temporary / "outputs"),
                    "--figure-dir", str(temporary / "figures"),
                ],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (temporary / "outputs" / "counts.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))

        # Complete outputs catch premature exits even if the CLI returns zero.
        self.assertEqual(len(rows), 2 * 4 * 18)
        self.assertEqual(
            {(int(row["simu_id"]), int(row["step"]), int(row["population_id"]))
             for row in rows},
            {(run, step, herd) for run in range(2)
             for step in range(4) for herd in range(18)},
        )
        assignments = {
            "A": {0}, "G": {1}, "F": {2}, "Jnb": {2}, "Jn": {3},
            "Jf": set(range(4, 16)),
        }
        for row in rows:
            with self.subTest(run=row["simu_id"], step=row["step"],
                              herd=row["population_id"]):
                herd = int(row["population_id"])
                for state in ("Female", "Male", "M", "S", "E", "I", "R"):
                    self.assertGreaterEqual(int(row[state]), 0)
                self.assertEqual(
                    sum(int(row[state]) for state in ("M", "S", "E", "I", "R")),
                    int(row["Female"]) + int(row["Male"]),
                )
                for age_group, allowed_herds in assignments.items():
                    count = int(row[age_group])
                    self.assertGreaterEqual(count, 0)
                    if herd not in allowed_herds:
                        self.assertEqual(count, 0, age_group)
                if herd == 4 and int(row["step"]) > 0:
                    sampled = int(float(row["nb_of_sampled_I_Jf"]))
                    self.assertGreaterEqual(sampled, 0)
                    self.assertLessEqual(sampled, 12)


if __name__ == "__main__":
    unittest.main()
