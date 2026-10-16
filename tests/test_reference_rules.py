"""Small, hand-calculated examples for article equations 5 and 6."""

import unittest
from pathlib import Path
from types import SimpleNamespace

from emulsion.model.emulsion_model import EmulsionModel

from movement_12fatpens_outrans import Metapopulation as Default
from movement_12fatpens_swIAVs import Metapopulation as Weighted
from test_transition_weights import Partition, manager


def weight(partitions, index, name="trans_btwn_pens_frm_movement"):
    return getattr(partitions[index].statevars, name)


def auxiliary(partitions):
    process = manager(partitions)
    process.model.parameters.third_path_trans = 2
    Weighted.third_trans_pathway(process)
    return [weight(partitions, i, "trans_frm_3rd_path") for i in range(16)]


class ReferenceRulesTest(unittest.TestCase):
    def test_compiled_yaml_local_rule_handles_empty_and_nonempty_partitions(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ('swine_model_weekly-12fatpens-FINAL.yaml',
                         'swine_model_weekly-12fatpens-weight-FINAL-swIAVs.yaml'):
            model = EmulsionModel(str(root / filename),
                                  changed_values={'transmission_rate': 2.0})
            rule = model.get_value('force_of_infection_int')
            for selected, size, expected in [(0, 0, 0), (0, 8, 0),
                                              (2, 8, 0.5), (8, 8, 2)]:
                with self.subTest(filename=filename, selected=selected, size=size):
                    values = {'total_I': selected, 'total_herd': size,
                              'transmission_rate': 2.0}
                    partition = SimpleNamespace(get_information=values.__getitem__)
                    self.assertAlmostEqual(rule(partition), expected)

    def test_neighbour_rules_sum_ratios_for_unequal_and_equal_sizes(self):
        for model in (Default, Weighted):
            for sizes, selected, expected in [((4, 12), (1, 6), 8.5),
                                               ((8, 8), (2, 4), 8.5)]:
                with self.subTest(model=model.__module__, sizes=sizes):
                    p = [Partition(size=0, weight=7) for _ in range(18)]
                    p[8] = Partition(size=8, weight=7)
                    p[6] = Partition(size=sizes[0], selected=selected[0])
                    p[10] = Partition(size=sizes[1], selected=selected[1])
                    model.sample_I_from_fatteners(manager(p))
                    self.assertAlmostEqual(weight(p, 8), expected)

    def test_default_focal_accumulators_are_independent(self):
        p = [Partition() for _ in range(18)]
        p[6].total_I = 4
        Default.sample_I_from_fatteners(manager(p))
        self.assertEqual(weight(p, 4), 1)
        self.assertEqual(weight(p, 5), 0)
        self.assertEqual(weight(p, 8), 1)
        self.assertEqual(weight(p, 10), 0)

    def test_connectivity_is_preserved_for_each_variant(self):
        for model, expected in ((Default, 0), (Weighted, 1)):
            with self.subTest(model=model.__module__):
                p = [Partition() for _ in range(18)]
                p[10].total_I = 4
                model.sample_I_from_fatteners(manager(p))
                self.assertEqual(weight(p, 4), expected)

    def test_empty_sources_contribute_zero(self):
        for model in (Default, Weighted):
            with self.subTest(model=model.__module__):
                p = [Partition(size=0, weight=3) for _ in range(18)]
                model.sample_I_from_fatteners(manager(p))
                self.assertTrue(all(weight(p, i) == 3 for i in range(18)))
        self.assertEqual(auxiliary([Partition(size=0) for _ in range(18)]), [0]*16)

    def test_auxiliary_sums_area_ratios_and_excludes_entire_focal_area(self):
        p = [Partition() for _ in range(18)]
        p[1] = Partition(size=4, selected=1)
        p[2] = Partition(size=12, selected=6)
        p[5].total_I = 3
        # Composite area: I=3, N=12*8. Auxiliary weight beta=2.
        actual = auxiliary(p)
        self.assertAlmostEqual(actual[0], 2*(1/4 + 6/12 + 3/96))
        self.assertAlmostEqual(actual[1], 2*(6/12 + 3/96))
        self.assertAlmostEqual(actual[2], 2*(1/4 + 3/96))
        for i in range(4, 16):
            self.assertAlmostEqual(actual[i], 2*(1/4 + 6/12))

    def test_every_composite_member_contributes_as_a_source(self):
        for index in range(4, 16):
            with self.subTest(index=index):
                p = [Partition() for _ in range(18)]
                p[index].total_I = 1
                # Non-area partitions 16 and 17 must be ignored.
                p[16].total_I = p[17].total_I = 8
                actual = auxiliary(p)
                for focal in range(4):
                    self.assertAlmostEqual(actual[focal], 2/96)
                self.assertEqual(actual[4:], [0]*12)

    def test_auxiliary_is_invariant_to_source_and_composite_member_permutations(self):
        p = [Partition() for _ in range(18)]
        p[1] = Partition(size=4, selected=1)
        p[2] = Partition(size=12, selected=6)
        p[4] = Partition(size=4, selected=2)
        p[5] = Partition(size=16, selected=1)
        before = auxiliary(p)
        p[1], p[2] = p[2], p[1]
        p[4], p[5] = p[5], p[4]
        after = auxiliary(p)
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[2])
        self.assertAlmostEqual(before[2], after[1])
        for i in range(3, 16):
            self.assertAlmostEqual(before[i], after[i])

    def test_neighbour_graph_is_invariant_under_graph_preserving_row_swap(self):
        for model in (Default, Weighted):
            with self.subTest(model=model.__module__):
                def partitions(swapped=False):
                    p = [Partition() for _ in range(18)]
                    p[6] = Partition(size=4, selected=1)
                    p[13] = Partition(size=12, selected=6)
                    if swapped:
                        for i in range(4, 16, 2):
                            p[i], p[i+1] = p[i+1], p[i]
                    return p
                left, right = partitions(), partitions(swapped=True)
                model.sample_I_from_fatteners(manager(left))
                model.sample_I_from_fatteners(manager(right))
                for i in range(4, 16, 2):
                    self.assertAlmostEqual(weight(left, i), weight(right, i+1))
                    self.assertAlmostEqual(weight(left, i+1), weight(right, i))

    def test_invalid_counts_fail_instead_of_generating_invalid_weights(self):
        for selected, size in [(1, 0), (-1, 8), (9, 8), (float('nan'), 8),
                                (0, float('inf')), (0, -1)]:
            for model in (Default, Weighted):
                with self.subTest(selected=selected, size=size, model=model.__module__):
                    p = [Partition() for _ in range(18)]
                    p[6] = Partition(size=size, selected=selected)
                    with self.assertRaises(ValueError):
                        model.sample_I_from_fatteners(manager(p))
                    with self.assertRaises(ValueError):
                        auxiliary(p)
