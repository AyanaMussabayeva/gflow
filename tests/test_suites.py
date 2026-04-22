from pkg.experiments.suites import build_suite_specs
from pkg.experiments.specs import make_experiment_spec


def test_main_suite_contains_three_benchmarks():
    specs = build_suite_specs("main_v2", "smoke")
    assert [spec.benchmark_name for spec in specs] == ["branin", "hartmann6", "ackley10"]


def test_pool_ablation_contains_shared_and_fresh():
    specs = build_suite_specs("pool_ablation_v2", "smoke")
    names = sorted({spec.name for spec in specs})
    assert names == ["pool_fresh_v2", "pool_shared_v2"]


def test_finetune_ablation_contains_continual_and_restart():
    specs = build_suite_specs("finetune_ablation_v2", "smoke")
    names = sorted({spec.name for spec in specs})
    assert names == ["finetune_continual_v2", "finetune_restart_v2"]


def test_smoke_profile_loaded_from_configs():
    spec = make_experiment_spec("branin", "main_v2", profile="smoke")
    assert spec.n_iter == 4
    assert spec.n_candidates == 256
