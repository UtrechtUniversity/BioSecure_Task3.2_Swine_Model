# Follow-up: two transition-weight implementation fixes

This branch builds on PR #2, baseline commit
`544ec0fa2ca178cb5858a3fb175efa5ba27b02a2`, without changing that PR's branch.
It addresses two implementation defects tracked in issue #3. Symbols and source
variable names here denote abstract partitions, discrete states, and numerical
weights only.

## Reference alignment

Article equation 2 composes the local, neighbour, schedule, and auxiliary terms
by addition. An update of one term must not erase an already accumulated term.
Equation 3 aggregates schedule contributions for the intended destination area.
The add-ons already expand destination area 4 into partitions 4 through 15;
the loop must therefore update each of those partition indices once.

The published reference is unchanged. These corrections align those two
implementation behaviours with its destination and additive-composition rules.
They do not establish full equivalence with every article equation.

## Changes and hand-calculated cases

| Case | Before | Corrected result |
| --- | --- | --- |
| Two schedule entries to area 4, with weights 4*(5040/10080)*(2/8) and 4*(2520/10080)*(4/8) | Partition 4 receives 12; partitions 5..15 receive 0. | Every partition 4..15 receives 1 exactly once. The same correction is applied to both add-ons. |
| A pen partition has existing schedule weight 7 and the neighbour contribution is 2*3/86 | The assignment leaves only 6/86, erasing 7. | The accumulator contains 7 + 6/86. |
| Neighbour contribution is zero, including an empty neighbour collection | The assignment erases any earlier weight. | The earlier weight is preserved. |

The second fix changes `=` to `+=` in the swIAVs add-on. The schedule process
already resets the accumulator once at the beginning of each model step. The
normal process order then adds schedule and pen contributions exactly once;
this fix does not add an additional reset or duplicate the process call.

## Validation

Four focused tests cover destination expansion and accumulation in both
variants, additive preservation, zero-count boundaries, and permutation of
other partitions. These tests fail on the frozen baseline (five failing
subcases across four tests) and pass with the changes. They exercise the actual
process methods with small, hand-constructed numerical partitions.

The full suite passes ten tests. Each model variant also completes one full
170-step simulation with 3,078 output rows and an HTML plot. Output checks cover
complete time/partition grids, nonnegative state counts, partition assignments,
and state-total consistency. These execution checks are not proof of exact
agreement with the published numerical sequences.

The initial hosted notebook check reached the sampling cell but exceeded its
1,200-second execution timeout. CI now allows 3,600 seconds per cell and
75 minutes for the whole job, including installation and tests. The documented
noninteractive command uses the same per-cell limit. These are finite execution
limits; notebook settings and numerical rules are unchanged by this CI adjustment.
The hosted check at revision `c5374c0` subsequently passed all steps, including
every notebook cell: <https://github.com/jmtubay1983/BioSecure_Task3.2_Swine_Model/actions/runs/38951472286>.

## Remaining review in issue #3

At the PR #4 baseline, the pooled-versus-sum-of-ratios choice, default running
neighbour accumulator, and auxiliary aggregation/connectivity were unchanged.
The focused test used the existing pooled expression to isolate additive
composition. The separate [reference-rule follow-up](reference-rule-fixes.md)
addresses those remaining calculations and updates that test's expected ratio
sum. It preserves the original variant-specific connectivity.

Computed outputs can change because the old implementation lost or misplaced
weights. EMULSION's object-identity-dependent agent ordering also prevents exact
sequence replay from seeds alone. Preserve baseline revision and actual saved
sequences; do not describe changed outputs as recovered published results.
