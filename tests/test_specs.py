from pkg.experiments import make_experiment_spec


def test_smoke_spec_uses_new_artifact_friendly_defaults():
    spec = make_experiment_spec("branin", name="main_v2", profile="smoke")
    assert spec.use_block_mask_training is True
    assert spec.reward_spec.protocol == "rank"
    assert spec.n_iter == 4
