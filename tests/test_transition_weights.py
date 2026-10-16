"""Hand-calculated checks for abstract schedule and partition weight composition."""

import importlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Partition:
    def __init__(self, size=8, selected=0, weight=0):
        self.total_herd = size
        self.total_I = selected
        self.total_Jnb = 0
        self.statevars = SimpleNamespace(
            trans_btwn_pens_frm_movement=weight,
            nb_of_sampled_I_Jf=0,
        )

    def select_atoms(self, *args):
        return []

    def remove_atoms(self, atoms):
        pass

    def add_atoms(self, atoms):
        pass


def manager(partitions, moves=None, neighbour_weight=2):
    return SimpleNamespace(
        get_populations=lambda: partitions,
        simulation=SimpleNamespace(shared_data={"moves": moves or {}}),
        statevars=SimpleNamespace(step=1),
        model=SimpleNamespace(parameters=SimpleNamespace(
            nb_herds=18, vert_trans=0, pba=0,
            max_bp_trans=1, min_bp_trans=0, mean_bp_trans=0.5,
            between_herd_trans=4, biosec_remove_risky_move=0,
            neighboring_herd_trans=neighbour_weight,
        )),
    )


class TransitionWeightTest(unittest.TestCase):
    def test_schedule_source_uses_the_entire_composite_area_even_if_partition_4_is_empty(self):
        for name, method in [
            ("movement_12fatpens_outrans", "animal_farmer_movement"),
            ("movement_12fatpens_swIAVs", "farmer_movement"),
        ]:
            for source_member in (None, 4, 5, 15):
                with self.subTest(variant=name, source_member=source_member):
                    module = importlib.import_module(name)
                    partitions = [Partition(size=0, weight=99) for _ in range(18)]
                    if source_member is not None:
                        partitions[source_member] = Partition(size=8, selected=2)
                    moves = {1: {4: [[3, 10080]]}}
                    with patch.object(module, "random_gamma", return_value=1):
                        getattr(module.Metapopulation, method)(manager(partitions, moves))
                    # One area ratio: 2/8 regardless of which member holds it.
                    expected = 0 if source_member is None else 1
                    self.assertAlmostEqual(
                        partitions[3].statevars.trans_btwn_pens_frm_movement, expected
                    )

    def test_schedule_weight_reaches_each_target_once_and_accumulates_entries(self):
        for name, method in [
            ("movement_12fatpens_outrans", "animal_farmer_movement"),
            ("movement_12fatpens_swIAVs", "farmer_movement"),
        ]:
            with self.subTest(variant=name):
                module = importlib.import_module(name)
                partitions = [Partition(weight=99) for _ in range(18)]
                partitions[0].total_I = 2
                partitions[1].total_I = 4
                partitions[2].total_I = 2
                moves = {1: {0: [[4, 5040], [0, 10080]],
                             1: [[4, 2520]], 2: [[3, 10080]]}}
                process = manager(partitions, moves)
                with patch.object(module, "random_gamma", return_value=1):
                    getattr(module.Metapopulation, method)(process)
                # Each pen receives 4*(5040/10080)*(2/8)
                #                       + 4*(2520/10080)*(4/8) = 1.
                for index in range(4, 16):
                    self.assertAlmostEqual(
                        partitions[index].statevars.trans_btwn_pens_frm_movement, 1
                    )
                # Direct destination 3 receives 4*(10080/10080)*(2/8) = 1.
                self.assertAlmostEqual(partitions[3].statevars.trans_btwn_pens_frm_movement, 1)
                for index in (0, 1, 2, 16, 17):
                    self.assertEqual(partitions[index].statevars.trans_btwn_pens_frm_movement, 0)

    def test_swiavs_partition_weight_is_added_to_existing_schedule_weight(self):
        from movement_12fatpens_swIAVs import Metapopulation

        partitions = [Partition(weight=7) for _ in range(18)]
        partitions[4] = Partition(size=4, selected=2, weight=7)
        partitions[5] = Partition(size=6, selected=3, weight=7)
        process = manager(partitions)
        Metapopulation.sample_I_from_fatteners(process)
        # Equation 5 adds per-source ratios, preserving the schedule term.
        self.assertAlmostEqual(partitions[4].statevars.trans_btwn_pens_frm_movement, 7 + 2*3/6)
        self.assertAlmostEqual(partitions[5].statevars.trans_btwn_pens_frm_movement, 7 + 2*2/4)
        self.assertAlmostEqual(partitions[6].statevars.trans_btwn_pens_frm_movement, 7 + 2*(2/4 + 3/6))
        self.assertEqual(partitions[4].statevars.nb_of_sampled_I_Jf, 2)
        for index in (0, 1, 2, 3, 16, 17):
            self.assertEqual(partitions[index].statevars.trans_btwn_pens_frm_movement, 7)

    def test_swiavs_zero_contribution_preserves_existing_weight(self):
        from movement_12fatpens_swIAVs import Metapopulation

        for size in (0, 8):
            with self.subTest(size=size):
                partitions = [Partition(size=size, weight=3) for _ in range(18)]
                Metapopulation.sample_I_from_fatteners(manager(partitions))
                for partition in partitions:
                    self.assertEqual(partition.statevars.trans_btwn_pens_frm_movement, 3)

    def test_swiavs_weight_is_invariant_to_order_of_other_partitions(self):
        from movement_12fatpens_swIAVs import Metapopulation

        partitions = [Partition() for _ in range(18)]
        partitions[4].statevars.trans_btwn_pens_frm_movement = 5
        partitions[5] = Partition(size=4, selected=1)
        partitions[6] = Partition(size=12, selected=6)
        Metapopulation.sample_I_from_fatteners(manager(partitions))
        expected = partitions[4].statevars.trans_btwn_pens_frm_movement
        partitions[5], partitions[6] = partitions[6], partitions[5]
        partitions[4].statevars.trans_btwn_pens_frm_movement = 5
        Metapopulation.sample_I_from_fatteners(manager(partitions))
        self.assertAlmostEqual(partitions[4].statevars.trans_btwn_pens_frm_movement, expected)
