from collections import Counter

from pcf.jit.jev_ood_benchmark import hard_jev_benchmark_cases
from pcf.jit.types import ExecutionPrimitive


def test_hard_jev_benchmark_is_balanced_and_unique() -> None:
    cases = hard_jev_benchmark_cases()
    assert len(cases) == 20
    assert len({case.name for case in cases}) == len(cases)
    counts = Counter(case.expected for case in cases)
    assert counts == {primitive: 4 for primitive in ExecutionPrimitive}


def test_hard_jev_benchmark_marks_all_cases_ood_boundary() -> None:
    for case in hard_jev_benchmark_cases():
        assert "ood" in case.tags
        assert "boundary" in case.tags
        assert case.state
