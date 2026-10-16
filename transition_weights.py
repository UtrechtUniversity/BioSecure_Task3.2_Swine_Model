"""State-count ratios for the numerical rules in article equations 5 and 6."""

from math import fsum, isfinite


PEN_INDICES = tuple(range(4, 16))
AREA_PARTITIONS = ((0,), (1,), (2,), (3,), PEN_INDICES)


def _counts(partition):
    selected, size = float(partition.total_I), float(partition.total_herd)
    if (not isfinite(selected) or not isfinite(size)
            or not 0 <= selected <= size):
        raise ValueError("State counts must be finite and satisfy 0 <= I <= N")
    return selected, size


def sum_state_ratios(partitions, indices):
    """Sum source-specific I/N values; a valid empty source contributes zero."""
    ratios = []
    for index in indices:
        selected, size = _counts(partitions[index])
        ratios.append(selected / size if size else 0.0)
    return fsum(ratios)


def pooled_state_ratio(partitions, indices):
    """Compute one area's I/N by pooling counts across its member partitions."""
    counts = [_counts(partitions[index]) for index in indices]
    size = fsum(size for _, size in counts)
    return fsum(selected for selected, _ in counts) / size if size else 0.0
