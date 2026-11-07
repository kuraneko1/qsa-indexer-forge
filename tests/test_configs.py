from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = [
    ROOT / "configs/text-local.yaml",
    ROOT / "configs/multimodal-local.yaml",
    ROOT / "configs/multimodal-continue-mm2.yaml",
]


def test_public_training_configs_are_local_and_pinned():
    for path in PUBLIC:
        cfg = yaml.safe_load(path.read_text())
        assert cfg["push_to_hub"] is False, path
        assert len(cfg["model_revision"]) == 40, path
        assert len(cfg["dataset_revision"]) == 40, path
        assert cfg["block_size"] == 4 and cfg["budget"] == 2048, path


def test_public_mm2_checkpoint_is_immutable():
    cfg = yaml.safe_load((ROOT / "configs/multimodal-continue-mm2.yaml").read_text())
    assert cfg["indexer_file"] == "indexers/mm2_16k/qsa_indexers.pt"
    assert len(cfg["indexer_revision"]) == 40
    assert len(cfg["indexer_sha256"]) == 64


def test_p0_config_is_dense_only_and_pinned():
    cfg = yaml.safe_load((ROOT / "configs/p0.yaml").read_text())
    assert "stage" not in cfg or cfg["stage"] == "capture"
    assert "indexer_repo" not in cfg and "push_to_hub" not in cfg
    assert cfg["budgets"] == [512, 1024, 2048, 4096]
    assert cfg["block_size"] == 4
    assert len(cfg["model_revision"]) == 40
    assert len(cfg["dataset_revision"]) == 40


def test_mm0_config_pins_the_published_indexers():
    cfg = yaml.safe_load((ROOT / "configs/mm0.yaml").read_text())
    assert cfg["stage"] == "mm0"
    assert cfg["indexer_file"] == "indexers/mm2_16k/qsa_indexers.pt"
    assert len(cfg["indexer_revision"]) == 40
    assert len(cfg["indexer_sha256"]) == 64
    assert cfg["block_size"] == 4 and cfg["budget"] == 2048
