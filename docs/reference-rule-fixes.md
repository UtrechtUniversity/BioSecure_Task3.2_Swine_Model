# Reference-rule aggregation and empty-state fixes

This follow-up builds on PR #4 at `c5374c0` and addresses the remaining
aggregation items in issue #3. PR #2's branch is unchanged. The equations in
the published article are the reference and are not edited:
<https://doi.org/10.1016/j.atech.2025.101652>.

## Implemented rules

Let `I[j]` be a selected-state count and `N[j]` a partition size. For each
nonempty partition, `q[j] = I[j]/N[j]`. A valid empty partition has `I=N=0`
and contributes zero. The neighbour/auxiliary ratio helpers reject negative,
nonfinite, or inconsistent counts (`I>N`) with `ValueError` rather than
generating an invalid numerical weight.

Equation 5 is evaluated separately for each focal partition:

`neighbour_weight[i] = beta_pen * sum(q[j] for j in J(i))`.

The existing connectivity variants are preserved:

- Default: two rows `(4,6,8,10,12,14)` and `(5,7,9,11,13,15)`; immediate
  neighbours differ by two, with one neighbour at each row endpoint.
- swIAVs: every other partition in 4–15, excluding the focal partition.

The result is added to the schedule contribution already present in the
accumulator. There is no accumulator shared across focal partitions.

Equation 6 is evaluated at area level. The explicit membership map is
`{0}`, `{1}`, `{2}`, `{3}`, and `{4,...,15}`. Counts are pooled *within* an
area to obtain its single ratio, then the ratios of other areas are summed:

`Q[a] = sum(I[j] for j in area[a]) / sum(N[j] for j in area[a])`

`auxiliary_weight[a] = beta_aux * sum(Q[b] for b != a)`.

An empty area contributes zero. Every member of the focal area receives the
same auxiliary weight, excluding its entire own area from the source sum.
Partitions 16 and 17 are outside this map. Pooling member counts within an area
does not replace the sum of source-area ratios with a pooled ratio across areas.

The map follows the repository's assignment of partitions 4–15 to one area,
its existing pooled schedule-source calculation for that area, and the main
article's area-level equation 6. Figure 3 permits both adjacency and expanded
pairwise connectivity for equation 5. The supplementary algorithms could not
be retrieved because the publisher hosts were blocked in this environment.
Supplementary confirmation of this explicit mapping remains outstanding; this
change does not claim complete equivalence with the original study code.

## Local and schedule boundary fixes

The two YAML local expressions now use
`transmission_rate * DIV(total_I, total_herd)`, using EMULSION's existing
zero-safe division function. For nonempty valid partitions this is exactly
equation 4. For an empty partition it returns zero instead of raising
`ZeroDivisionError`. The regression test loads each actual YAML through
`EmulsionModel` and evaluates the compiled expression for empty and nonempty
counts; it does not substitute a separately written numerical formula.

Schedule source 4 represents the whole composite area. Both add-ons now
evaluate its pooled state ratio even when partition 4 itself is empty, rather
than skipping a nonempty area based on that one member's size. The existing
area-level ratio and normalized schedule weight are preserved. Tests place the
same counts in members 4, 5, or 15 and require the same contribution, and verify
that an entirely empty area contributes zero.

## Hand-calculated checks

| Case | Baseline behaviour | Revised result |
| --- | --- | --- |
| Neighbour beta=2, two nonempty sources `(I,N)=(1,4),(6,12)` | Pooled source ratio gives `0.875` in swIAVs. | `2*(1/4+6/12)=1.5`, before any existing schedule contribution. |
| Default beta=2, all sizes 8, only partition 6 has count 4 | Focal 5 receives 1 carried from focal 4. | Focal 4 receives 1; focal 5 receives 0 from its own source set. |
| Auxiliary beta=2, areas 1 and 2 have `(1,4),(6,12)`, composite area has `I=3,N=96` | Cumulative numerator is divided by the last source size. | Area 0 receives `2*(1/4+6/12+3/96)=1.5625`. |
| Only one composite member has count 1, all member sizes 8, beta=2 | Members 5–15 are ignored as sources. | Each other area receives `2/96`; the composite area receives zero from itself. |
| All sources empty | Default neighbour method divides by zero. | Zero additional contribution; prior schedule weight is preserved. |

Nine aggregation tests exercise actual process methods, including unequal and equal
source sizes, independent focal accumulators, connectivity, empty sources,
invalid counts, every composite-area member, source-area permutations, member
permutations, and a graph-preserving swap of the two default rows. They exposed
30 failing assertions/subcases and four division-by-zero errors before repair.
Two additional boundary tests exposed four schedule subcase failures and two
compiled-YAML division-by-zero errors before their corresponding fixes. The
complete suite contains 21 tests.
The existing additive-composition test now uses equation 5's source-specific
ratios rather than asserting the old pooled expression.

## Output comparison and limits

The baseline is frozen at `c5374c0`; its source files and output CSVs are retained
separately from revised runs. Each variant is checked over 170 steps using seed
2026. Validation checks the complete time/partition grid, nonnegative counts,
state totals, permitted partition assignments, and plot generation.

EMULSION's object-identity-dependent ordering prevents exact sequence replay
from seeds alone. A difference between two generated sequences cannot therefore
be attributed solely to the formula changes. The hand-calculated method tests
establish the specific numerical corrections. Full-run outputs establish
execution and structural invariants, not recovery of the published results.

Saved observations must be reused explicitly for any later estimation
comparison. The notebook still generates new synthetic observations by default;
it does not reuse the original study's unavailable sequences. Parameter priors
and generation budgets are unchanged. The YAML local expression changes only
its valid empty-partition behaviour; its nonempty formula is preserved.

## Why issue #3 remains open

The formula defects covered here are implemented and tested, but supplementary
confirmation of the area mapping and independent model evaluation remain.
No issue-closing keyword should be used in this PR.

The local and schedule empty-state checks described above are now covered by
this same follow-up. Passing them does not establish complete validation of
every state transition or recovery of the original published sequences.
