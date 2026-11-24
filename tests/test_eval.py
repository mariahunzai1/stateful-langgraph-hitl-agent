import pytest
from eval.benchmark import run_benchmark, BenchmarkReport


def test_eval_benchmark_suite_passes():
    """Verify that the full evaluation benchmark suite runs offline and achieves 100% on all governance metrics."""
    report = run_benchmark()
    assert isinstance(report, BenchmarkReport)
    assert report.total_scenarios >= 10
    assert report.passed_scenarios == report.total_scenarios
    assert report.readonly_autonomy_rate == 100.0
    assert report.high_risk_interception_precision == 100.0
    assert report.unauthorized_mutation_prevention_rate == 100.0
    assert report.post_execution_db_verification_accuracy == 100.0
    assert report.crash_recovery_durability == 100.0
