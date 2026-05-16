from pathlib import Path


def test_common_make_run_commands_enable_early_stopping():
    makefile = Path("Makefile").read_text()

    assert "AUTO_STOP_ARGS :=" in makefile
    assert "REAL_AUTO_STOP_ARGS :=" in makefile
    assert "REAL_ADAPTIVE_ARGS :=" in makefile
    assert "REAL_MAX_TRAIN_SAMPLES ?= 2400" in makefile
    assert "REAL_MAX_TEST_SAMPLES ?= 600" in makefile
    assert "NODULE_MAX_TRAIN_SAMPLES ?= 800" in makefile
    assert "NODULE_MAX_TEST_SAMPLES ?= 240" in makefile
    assert "--early-stopping" in makefile
    assert "--adaptive-rounds" in makefile
    assert "REAL_TARGET_ARGS ?=" in makefile
    assert "demo:" in makefile
    assert "run:" in makefile
    assert "run-real:" in makefile
    assert "run-real-2d:" in makefile
    assert "run-real-3d:" in makefile
    assert "uv run flxbc demo --synthetic" in makefile
    assert "uv run --extra app flxbc run --synthetic" in makefile
    assert "uv run --extra app --extra ml flxbc run --dataset pneumoniamnist" in makefile
    assert "uv run --extra app --extra ml flxbc run --dataset nodulemnist3d" in makefile
    assert "--max-train-samples $(REAL_MAX_TRAIN_SAMPLES)" in makefile
    assert "--max-test-samples $(REAL_MAX_TEST_SAMPLES)" in makefile
