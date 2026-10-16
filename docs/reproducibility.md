# Computational workflow and reproducibility review

Reference: Tubay et al., *HogDSim - A within-farm modeling framework for infectious
diseases of swine*, Smart Agricultural Technology 12 (2025), 101652.
<https://doi.org/10.1016/j.atech.2025.101652>.

This review treats agents, state labels, transition weights, and pen/area labels
as abstract elements of a discrete computational system. Existing variable names
are retained to map the review to the source files. In particular, `I` denotes a
selected discrete state; `nb_I` in the notebook's CSV schema counts partitions
containing at least one agent in that state.

## Execution defects and repairs

- The published default add-on executed movement statements at module scope,
  where `herds` was undefined. Process methods were outside `Metapopulation`,
  and the collection of partitions was incorrectly used as a single partition.
  The repair restores method nesting and explicit partition indices from
  personal-repository commit `59b4355` (`jmtubay1983/Swine-Model`), keeping the
  current YAML's `animal_farmer_movement` method name.
- The original notebook referenced an absent `SWINE_MODEL.yaml`. The adapter
  explicitly selects one of the two YAML files actually in the repository and
  resolves all input/add-on paths relative to its own location.
- The original notebook unconditionally read an absent `infected_data.csv`,
  then overwrote the selected observations with generated sequences. The new
  workflow explicitly selects generated or supplied observations; it does not
  silently substitute one for the other.
- The original unused Redis address is removed. The example uses a local serial
  sampler and SQLite. Each simulation gets temporary outputs, validates a full
  time/partition grid, and reports failures before ABC consumes a sequence.
- The test file uploaded at the repository root is moved to `tests/`, matching
  the documented discovery command and its relative path calculation.
- The resolved dependency set pins a compatible notebook display stack.
  Matplotlib 3.8.4 with IPython 9.17.1 passed package metadata checks but failed
  at runtime (`backend2gui` import). IPython 8.37.0, ipykernel 6.29.5, and
  matplotlib-inline 0.1.7 address that observed incompatibility.

These repairs restore execution and provide explicit data handling. They do not
change the mathematical weight formulas in either YAML or add-on.

## Supplied inputs versus synthetic outputs

| Component | Provenance and role |
| --- | --- |
| `data/modified_movements_10yr.csv` | Supplied schedule input with `date,source,dest,duration`. The article describes the underlying schedule records as processed and anonymized empirical inputs. The exact modification/extension history of this public file has not been established; it should not be described as an unmodified raw dataset. The notebook does not generate it. |
| Original missing `infected_data.csv` | A missing precomputed observation input expected by the old notebook. It is not evidence that field state measurements were used: sections 2.3.2–2.3.3 explicitly describe synthetic state-count sequences for calibration. |
| New default observations | Newly generated sequences from the selected swIAVs configuration, explicitly labelled synthetic, with configuration, candidate seeds, selection index, and actual counts saved. They are not recovered original study outputs. |
| Optional user observations | An explicitly supplied CSV containing `week,nb_I`. The loader requires one row per week from 0 through the configured horizon and integer partition counts from 0 to 12. No silent interpolation or substitution is performed. |
| ABC particles and plots | Computed outputs of the example. They are not original published estimates or proof of parameter recovery. |

## Mapping the example to the article

The article's swIAVs configuration has four parameter weights: beta_int,
beta_ext, beta_pen, and beta_aux. The notebook maps these to `trans`,
`trans_betwn_herds`, `trans_neighbor`, and `trans_aux`; the last maps to YAML
`third_path_trans`. The legacy default configuration has only the first three
configurable weights in this adapter. A passing run of one variant does not
validate the other variant's rule set.

Sections 2.3.1 and 2.3.3 describe uniform prior ranges of 0..20, 0..15, 0..15,
and 0..1 per week; adaptive Manhattan distance; MedianEpsilon; 100 accepted
particles; a 0.001 stopping threshold; and up to 5 generations. Section 2.4 uses
a 170-step horizon. The candidate generating weights in section 2.3.3 are
17.01, 9.87, 9.87, and 0.1. The fourth YAML default is 17.01/178 (approximately
0.09556), so the notebook explicitly supplies 0.1 when generating its example.

The article generated 100 candidate sequences and selected an actual sequence
closest to their componentwise median. The notebook supports that selection
procedure through `SYNTHETIC_REPLICATES`; Euclidean closeness follows the original
notebook. It saves candidate sequences and the chosen index. Its quick defaults
use one candidate, 8 time steps, 12 particles, and at most 2 populations. It uses
the article's four prior ranges, adaptive Manhattan distance, MedianEpsilon,
and 0.001 threshold. PCMAD scaling follows the original notebook implementation;
the article describes adaptive scaling more generally.

Only the article's partition-count observation scenario is demonstrated. The
other observation granularities and the full expensive study configuration are
not reproduced. The notebook explicitly labels this limit. Increasing run sizes
alone does not resolve the mathematical discrepancies below.

## Mathematical rule-set discrepancies recorded at the PR #2 baseline

The table below records baseline commit `544ec0f`. On the separate follow-up
branch, destination expansion and additive preservation are corrected; see
[transition-weight fixes](transition-weight-fixes.md). The remaining ratio and
aggregation discrepancies are recorded here as historical findings.

See the [remaining issue #3 review](issue-3-rule-review.md) for executed
counterexamples, boundary findings, and the specification decisions required
before changing those rules.

The separate [reference-rule follow-up](reference-rule-fixes.md) implements
source-specific ratio sums, independent focal accumulators, and an explicit
composite-area map. It records the remaining reference-confirmation limits.

Let I_i count agents in state I within partition i, N_i be its total count, and
q_i = I_i/N_i for nonempty partitions. Let tau_f be a normalized schedule weight
and J(i) a configured neighbour set. No domain mechanism is inferred from these
symbols.

| Article formulation | Current implementation | Consequence and later review |
| --- | --- | --- |
| Eq. 4: gamma_int,i = beta_int * q_i | `force_of_infection_int` uses this expression in both YAML files. | The nonempty-partition algebra matches. Behaviour for empty partitions and execution order still needs separate boundary analysis. |
| Eq. 3: sum of entry weights beta_ext * tau_f * q_source(f) | Both add-ons accumulate schedule contributions. For destination 4, the loop over the other pen indices repeatedly updates `herds[dest]` instead of `herds[i]`. | The loop concentrates the replicated weight in partition 4 instead of distributing it across the intended partitions. Correcting it would change computed outputs. |
| Eq. 2: gamma_int + gamma_pen + gamma_ext + gamma_aux | In the swIAVs add-on, `farmer_movement` writes schedule contributions into `trans_btwn_pens_frm_movement`, then `sample_I_from_fatteners` assigns a pen contribution to the same variable. | Assignment overwrites the earlier contribution for pen partitions. Merely supplying beta_ext does not establish that this term reaches every intended partition. |
| Eq. 5: beta_pen * sum over J(i) of q_j | The swIAVs implementation uses a pooled ratio, sum(I_j)/sum(N_j), across other pen partitions. The default add-on uses pooled neighbour ratios and accumulates `trans_between_pens` across focal partitions. | A sum of ratios differs from a ratio of sums, and a running accumulation introduces partition-order dependence. The article's Fig. 3 allows different connectivity variants; the intended neighbour set must also be confirmed. |
| Eq. 6: beta_aux * sum over j != i of q_j | `third_trans_pathway` accumulates numerators, overwrites the denominator within the loop, and iterates over only the first five partition indices. | This is not the stated sum of per-partition ratios. The intended aggregation of the multiple pen partitions into the abstract area graph must be specified before correction. |

**Recommendation:** keep these formulas unchanged in the execution-repair PR and
open a separate rule-set evaluation PR. That later work should establish explicit
partition aggregation and connectivity, then compare implementations on small
hand-computable graphs, zero-count boundaries, partition-order permutations, and
route-composition cases. Record output changes against a frozen baseline rather
than presenting corrected numerical sequences as the original published results.
The operational indexing restoration in the default add-on is part of the
execution repair; the equation/weight changes above remain deferred.

## Reproducibility limits

The adapter initializes Python and NumPy random streams and restores the caller's
states so simulator calls do not reset ABC's sampling stream. However, EMULSION
1.2rc5's `AbstractAgent.__hash__` uses object identity (`hash(id(self))`). Agent
collections can therefore iterate differently across allocations, so identical
seeds do not guarantee identical state sequences. This was observed when a
longer repeat-seed check diverged even though the caller random-state checks
passed. The adapter does not patch the dependency to hide this behaviour.

Save and reuse the actual generated observation CSV, configuration, dependency
lock, selected YAML, source revision, and ABC history. Reproducible software
execution, exact numerical replay, recovery of generating weights, and agreement
with the article are distinct claims. The current PR addresses the first claim
and supplies artifacts for further evaluation; it does not establish the others.
