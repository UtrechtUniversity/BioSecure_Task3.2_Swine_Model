# Issue #3: remaining numerical-rule review

Reviewed source revision: `c5374c0a21c25e5d608b9def2666aef040b51989`.
Reference: <https://doi.org/10.1016/j.atech.2025.101652>, equations 5 and 6,
sections 2.1.5–2.1.6, and Figure 3. Source names below identify abstract
partitions, state counts, and weights. This review changes no numerical rules.

Issue #3 should remain open. The earlier destination-index and additive-update
fixes address only two of its items; passing execution checks does not establish
equivalence with the article. This document records the baseline review; the
separate [implementation follow-up](reference-rule-fixes.md) describes the
subsequent repairs and outstanding mapping confirmation.

## Confirmed discrepancies

### Sum of ratios versus pooled ratio

For a fixed source set J(i), equation 5 is
`beta * sum(I[j] / N[j] for j in J(i))`.
The swIAVs add-on, `sample_I_from_fatteners` at lines 295–305, computes
`beta * sum(I[j]) / sum(N[j])` over all other partitions 4–15 instead.
These are different expressions even when all nonempty sources have equal size:
pooling normalizes by their combined size, whereas equation 5 adds each ratio.
The default add-on also pools the two sources of an interior partition.

Executed example using the actual swIAVs method: set beta to 2, focal partition
4 to `(I,N)=(0,8)`, sources 5 and 6 to `(1,4)` and `(6,12)`, and all other
partitions to `(0,0)`. The implementation returns `2*7/16 = 0.875`.
The sum-of-ratios expression over the same two nonempty sources returns
`2*(1/4 + 6/12) = 1.5`. For this comparison only, empty sources contribute
zero; that convention must be stated explicitly in a future implementation.

### Default accumulator carries between focal partitions

In the default add-on, lines 281–297 initialize `trans_between_pens` before
the loop over focal partitions and repeatedly increment it. Each focal output
therefore includes terms evaluated for earlier focal indices, rather than just
its own source set. This contradicts a separate evaluation of equation 5 for
each focal index.

Executed example: all partition sizes are 8, all selected counts are zero except
partition 6, which has count 4, and beta is 2. Focal partition 4 correctly gets
1 from source 6. Focal partition 5 also gets 1, although its only source in the
implemented two-row graph is partition 7, whose selected count is zero. Its
reference contribution is zero. This demonstrates cross-focal carry without
changing the graph or relying on a stochastic simulation.

### Auxiliary denominator and composite-area representation

The swIAVs `third_trans_pathway`, lines 312–326, adds selected counts but
overwrites `total_herd` for each source. With nonempty sources, it divides their
accumulated selected count by the last visited source's size. Equation 6 instead
adds source-specific ratios. Merely changing `total_herd = ...` to `+= ...`
would produce a pooled ratio and would still not implement equation 6.

Executed denominator example: beta is 1, focal index is 0, and the first five
partitions have `(I,N)` values `(0,8), (1,4), (6,12), (0,8), (0,8)`.
The method returns `7/8 = 0.875`; equation 6 for these five singleton groups
returns `1/4 + 6/12 = 0.75`. Swapping source partitions 2 and 4 changes the
implementation's result to `7/12`, while the source-specific sum remains 0.75.
This permutation example isolates the denominator defect; it does not assert
that five singleton partitions are the intended composite-area mapping.

Both loops visit only indices 0–4. The resulting weight for focal index 4 is
copied to partitions 4–15, but source indices 5–15 are never read. The repository
maps those twelve partitions to one composite area. Thus reading partition 4
alone does not aggregate that area's state counts.

Executed omission example: all partitions have size 8, only partition 5 has
selected count 1, and beta is 1. Focal 0 receives zero. If the composite area's
ratio is defined using total counts across partitions 4–15, its contribution
would instead be `1/96`. This expected value is conditional on that explicit
aggregation rule, rather than a claim that the article specifies every detail
of the implementation mapping.

## Boundaries and specification decisions

- **Empty sources:** the default neighbour method raises `ZeroDivisionError`
  for an all-empty fixture. The swIAVs neighbour method completes with zero
  contribution. Equations 5 and 6 alone do not define `0/0`. Specify zero
  contribution for a valid empty source, and reject inconsistent counts such
  as positive selected count with zero size, before changing the formulas.
- **Connectivity:** the default code uses two rows, with indices differing by
  two as adjacent sources; the swIAVs code uses every other partition in 4–15.
  Figure 3 permits both adjacent and expanded pairwise connectivity. Neither
  graph is automatically an error. Preserve and explicitly name each variant's
  source set rather than silently selecting one graph for both.
- **Composite area:** equation 6 is stated at area level. A concrete candidate
  mapping is singleton areas `{0}`, `{1}`, `{2}`, `{3}`, plus area `{4,...,15}`,
  with each area ratio computed from its summed counts. Exclude the entire
  focal area, then apply its computed weight to its member partitions. Confirm
  this mapping against the supplementary algorithms or the authors' reference
  implementation before claiming full published-model equivalence.
- **Local expression:** both YAML files contain the literal `I/N` expression
  without an explicit empty-partition branch. This is a static observation;
  the diagnostic fixtures above do not test EMULSION's evaluation of that YAML
  expression, so an end-to-end empty-partition failure is not established here.

## Recommended follow-up

Preserve PR #2's execution scope and the existing follow-up's two narrow fixes.
Use a separate numerical-rule PR for per-focal reset, source-specific ratios,
and the explicit composite-area mapping. Keep equations 5 and 6 unchanged in
the reference; make code and tests conform to the selected interpretation.

Before implementation, record the graph and empty-source convention for each
variant and resolve the composite-area mapping. Then add hand-calculated tests
for unequal source sizes, equal source sizes, independent focal accumulators,
empty sources, graph-preserving relabelling, and selected counts confined to
each member of the composite area. For the adjacent graph, arbitrary index
swaps are not an invariance test unless the adjacency map is relabelled too.

Compare outputs to a saved source revision and saved observations. Treat new
sequences as outputs of the corrected implementation, not as recovered original
study sequences. The notebook's generated observations remain synthetic; the
supplied movement schedule is an input, not generated state-sequence data.

## Validation limits

The numerical examples above invoke the current process methods using the
small deterministic partition fixtures already provided in
`tests/test_transition_weights.py`. They demonstrate the recorded differences
without changing source files, notebook budgets, parameters, or dependency pins.
They are diagnostic examples, not a new passing regression suite for corrected
rules. No full calibration or claim of parameter recovery follows from them.

The prior hosted execution check for revision `c5374c0` passed installation,
dependency checks, the existing test suite, and every notebook cell:
<https://github.com/jmtubay1983/BioSecure_Task3.2_Swine_Model/actions/runs/38951472286>.
That workflow result verifies execution, not the mathematical equivalence
examined in this review.
